"""
Round 2 of the outer-loop horse-race: the methods that aim for BOTH a convergence
guarantee and practical speed, on the ill-conditioned (nonneg-box) instance.

Contenders, all minimizing the SAME debiased-FY loss over the SAME box {beta free,
gamma >= 0} from the same start, counting inner batched IPM solves (the cost unit):

  newton   -- projected damped Newton (current production)
  spg-bb   -- spectral projected gradient / Barzilai-Borwein (from bb_horserace)
  lbfgsb   -- scipy L-BFGS-B with our closed-form gradient (best-practice quasi-Newton;
              handles the box natively, exploits local curvature)
  ufgm     -- Nesterov Universal Fast Gradient Method (CORE 2013/26, eq (4.1)): the
              accelerated, smoothness-adaptive method with a provable rate under
              (M_nu, nu)-Holder gradients and NO knowledge of nu/M_nu.  Euclidean prox;
              the argmin steps are projections onto the box; line search with the
              eps/2 * tau slack that makes it work without a global Lipschitz constant.

We use the nonneg box (so L-BFGS-B applies directly and gamma>=0 keeps h''>=1, i.e. the
inner solve robust), at a moderate B for speed; the gamma-block is still collinear
(cond(A) ~ 1e4), so this is a genuine ill-conditioned test.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from bb_horserace import CASES, CountingSolver, _solver, spg_bb  # reuse harness
from estim_dgp import CATALOG, build_network, make_ods, make_problem
from scipy.optimize import BFGS, Bounds, minimize
from tr_bfgs import tr_bfgs  # lean hand-rolled trust-region BFGS

from purc.estimators.debiased_fy import (
    DebiasedFYEstimator,
    DebiasedFYLoss,
    EstimatorConfig,
    GammaProjection,
)
from purc.static_purc.dgp import simulate_dataset
from purc.static_purc.utils.torch_compat import DEFAULT_DTYPE, to_numpy


def _box_proj(layout, theta):
    """Project onto Q = {beta free, gamma >= 0} (the nonneg box)."""
    beta, g = layout.unpack(theta)
    return layout.pack(beta, torch.clamp(g, min=0.0))


def lbfgsb(loss, layout, theta0, n_beta, *, tol=1e-7, max_iter=3000):
    """Scipy L-BFGS-B on the box, with a memoized closed-form (value, grad)."""
    cache = {}

    def vg(x):
        key = x.tobytes()
        if key not in cache:
            th = torch.as_tensor(x, dtype=DEFAULT_DTYPE)
            Q, g = loss.value_and_grad(th)
            cache[key] = (float(Q), to_numpy(g).astype(np.float64))
        return cache[key]

    P = layout.size
    bounds = [(None, None)] * n_beta + [(0.0, None)] * (P - n_beta)
    res = minimize(
        lambda x: vg(x)[0], to_numpy(theta0).astype(np.float64), jac=lambda x: vg(x)[1],
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": tol, "maxfun": 10 * max_iter},
    )
    th = torch.as_tensor(res.x, dtype=DEFAULT_DTYPE)
    gmap = float((th - _box_proj(layout, th - torch.as_tensor(res.jac, dtype=DEFAULT_DTYPE))).abs().max())
    return dict(theta=th, n_outer=int(res.nit), converged=bool(res.success), Q=float(res.fun), gmap=gmap)


def trust_constr(loss, layout, theta0, n_beta, *, tol=1e-7, max_iter=400):
    """
    Trust-region quasi-Newton (scipy 'trust-constr') with our closed-form gradient and
    a BFGS Hessian *approximation* -- so NO finite differences, NO sensitivity dx*/dv,
    and NO line search (a ratio test governs the trust radius).  The nonneg box enters
    as simple bounds; the BFGS model is built from gradient secants only.
    """
    cache = {}

    def vg(x):
        key = x.tobytes()
        if key not in cache:
            Q, g = loss.value_and_grad(torch.as_tensor(x, dtype=DEFAULT_DTYPE))
            cache[key] = (float(Q), to_numpy(g).astype(np.float64))
        return cache[key]

    P = layout.size
    lb = np.array([-np.inf] * n_beta + [0.0] * (P - n_beta))
    ub = np.full(P, np.inf)
    res = minimize(
        lambda x: vg(x)[0], to_numpy(theta0).astype(np.float64), jac=lambda x: vg(x)[1],
        hess=BFGS(), method="trust-constr", bounds=Bounds(lb, ub),
        options={"maxiter": max_iter, "gtol": 1e-9, "xtol": 1e-14, "barrier_tol": 1e-10, "verbose": 0},
    )
    th = torch.as_tensor(res.x, dtype=DEFAULT_DTYPE)
    g = torch.as_tensor(vg(res.x)[1], dtype=DEFAULT_DTYPE)
    gmap = float((th - _box_proj(layout, th - g)).abs().max())
    return dict(theta=th, n_outer=int(res.nit), converged=bool(res.status in (1, 2)),
                Q=float(res.fun), gmap=gmap)


def ufgm(loss, layout, theta0, *, eps=1e-7, L0=1.0, tol=1e-7, max_iter=4000, max_bt=40):
    """
    Nesterov Universal Fast Gradient Method (CORE 2013/26, eq (4.1)), Euclidean prox,
    Q = nonneg box.  v_k, x_hat are projections; the line search tests the (4.1) descent
    with the eps/2 * tau slack.  Returns the same result dict as the others.
    """
    P = lambda th: _box_proj(layout, th)
    x0 = P(theta0.to(DEFAULT_DTYPE).reshape(-1))
    s = torch.zeros_like(x0)  # sum_j a_j grad f(x_j); v_k = P(x0 - s)
    A = 0.0
    y = x0.clone()
    L = L0
    nit, converged, gmap = 0, False, float("inf")
    for k in range(max_iter):
        nit = k + 1
        v = P(x0 - s)
        i = 0
        while True:  # backtracking line search over i_k (trial L = 2^i L)
            Lt = (2.0**i) * L
            a = (1.0 + math.sqrt(1.0 + 4.0 * Lt * A)) / (2.0 * Lt)  # a^2 Lt = A + a
            Anew = A + a
            tau = a / Anew
            xk1 = tau * v + (1.0 - tau) * y
            f_x, g_x = loss.value_and_grad(xk1)
            if not np.isfinite(f_x):  # off-domain trial: shrink step (raise L)
                i += 1
                if i > max_bt:
                    break
                continue
            xhat = P(v - a * g_x)
            yk1 = tau * xhat + (1.0 - tau) * y
            f_y = loss.value(yk1)
            rhs = (f_x + float(g_x @ (yk1 - xk1)) + (2.0 ** (i - 1)) * L * float(((yk1 - xk1) ** 2).sum())
                   + 0.5 * eps * tau)
            if np.isfinite(f_y) and f_y <= rhs + 1e-18:
                break
            i += 1
            if i > max_bt:
                break
        x, y, A, L = xk1, yk1, Anew, (2.0 ** (i - 1)) * L
        s = s + a * g_x
        _, g_y = loss.value_and_grad(y)  # stationarity probe (counted as a solve)
        gmap = float((y - P(y - g_y)).abs().max())
        if gmap < tol:
            converged = True
            break
    return dict(theta=y, n_outer=nit, converged=converged, Q=float(loss.value(y)), gmap=gmap)


def run(case, B, D):
    th0 = CATALOG[case["th0"]]
    inc, attrs = build_network(case["net"], case["nseed"])
    prob = make_problem(inc, th0, seed=case["pseed"], attrs=attrs)
    ods = make_ods(inc.shape[0], B, D, case["odseed"])
    base = _solver()
    base.preprocess(prob)
    data = simulate_dataset(prob, base, (th0.beta, th0.gamma), ods, np.random.default_rng(case["simseed"]))
    P = th0.beta.size + (th0.L - 2)
    th0_vec = np.concatenate([th0.beta, th0.gamma])

    def fresh_loss():
        s = CountingSolver(_solver())
        s.preprocess(prob)
        return DebiasedFYLoss(prob, s, data, th0.L, warm_start=True), s

    rows = []

    # newton (production)
    sN = CountingSolver(_solver())
    sN.preprocess(prob)
    est = DebiasedFYEstimator(prob, sN, th0.L,
                              EstimatorConfig(proj=GammaProjection("nonneg"), max_iter=300, tol_grad=1e-6))
    t0 = time.perf_counter()
    rN = est.fit(data)
    rows.append(("newton", rN.n_outer, sN.n_solve_batch, (time.perf_counter() - t0) * 1e3,
                 rN.converged, rN.objective, float(rN.grad.abs().max()), to_numpy(rN.theta_hat)))

    # spg-bb
    loss, s = fresh_loss()
    t0 = time.perf_counter()
    r = spg_bb(loss, GammaProjection("nonneg"), loss.layout, torch.zeros(P, dtype=DEFAULT_DTYPE),
               max_iter=4000, tol=1e-6)
    rows.append(("spg-bb", r["n_outer"], s.n_solve_batch, (time.perf_counter() - t0) * 1e3,
                 r["converged"], r["Q"], r["gmap"], to_numpy(r["theta"])))

    # lbfgsb (best practice)
    loss, s = fresh_loss()
    t0 = time.perf_counter()
    r = lbfgsb(loss, loss.layout, torch.zeros(P, dtype=DEFAULT_DTYPE), th0.beta.size, tol=1e-6)
    rows.append(("lbfgsb", r["n_outer"], s.n_solve_batch, (time.perf_counter() - t0) * 1e3,
                 r["converged"], r["Q"], r["gmap"], to_numpy(r["theta"])))

    # trust-constr (TR quasi-Newton via scipy: no FD, no sensitivity, no line search)
    loss, s = fresh_loss()
    t0 = time.perf_counter()
    r = trust_constr(loss, loss.layout, torch.zeros(P, dtype=DEFAULT_DTYPE), th0.beta.size, tol=1e-6)
    rows.append(("tr-qn(sp)", r["n_outer"], s.n_solve_batch, (time.perf_counter() - t0) * 1e3,
                 r["converged"], r["Q"], r["gmap"], to_numpy(r["theta"])))

    # tr-bfgs (lean hand-rolled trust-region BFGS, same nonneg box)
    loss, s = fresh_loss()

    def vg(x):
        Q, gg = loss.value_and_grad(torch.as_tensor(x, dtype=DEFAULT_DTYPE))
        return float(Q), to_numpy(gg).astype(np.float64)

    t0 = time.perf_counter()
    th, info = tr_bfgs(vg, np.zeros(P), th0.beta.size, bernstein_M=None, tol=1e-6)
    rows.append(("tr-bfgs", info["n_outer"], s.n_solve_batch, (time.perf_counter() - t0) * 1e3,
                 info["converged"], info["Q"], info["gmap"], th))

    # ufgm (universal fast gradient -- provable)
    loss, s = fresh_loss()
    t0 = time.perf_counter()
    r = ufgm(loss, loss.layout, torch.zeros(P, dtype=DEFAULT_DTYPE), eps=1e-7, tol=1e-6)
    rows.append(("ufgm", r["n_outer"], s.n_solve_batch, (time.perf_counter() - t0) * 1e3,
                 r["converged"], r["Q"], r["gmap"], to_numpy(r["theta"])))

    print(f"\n=== {case['name']} | {case['th0']} ({case['net']}) | nonneg box | B={B} D={D} | P={P} ===")
    print(f"  {'method':8s} {'conv':>5s} {'outer':>5s} {'solves':>7s} {'ms':>8s} {'Q':>13s} {'|gmap|':>9s} {'||th-th0||':>10s}")
    for name, outer, solves, ms, conv, Q, gm, th in rows:
        err = float(np.abs(th - th0_vec).max())
        print(f"  {name:8s} {str(conv):>5s} {outer:>5d} {solves:>7d} {ms:>8.0f} {Q:>13.6e} {gm:>9.1e} {err:>10.1e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=120)
    ap.add_argument("--D", type=int, default=500)
    ap.parse_args()
    a = ap.parse_args()
    ill = next(c for c in CASES if c["name"] == "ill_cond")
    run(ill, a.B, a.D)
