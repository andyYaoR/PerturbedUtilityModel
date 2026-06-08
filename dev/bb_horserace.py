"""
Horse-race: the estimator's outer loop -- projected damped Newton (current) vs a
Spectral Projected Gradient (SPG = projected Barzilai--Borwein + Grippo--Lampariello
--Lucidi nonmonotone line search; Birgin--Martinez--Raydan 2000).

Both optimizers minimize the *same* debiased-FY loss over the *same* convexity set,
from the same start, on the same simulated data.  We report, per (network, theta0,
basis, method): outer iterations, number of inner batched IPM solves (the real cost
unit -- one ``solve_batch`` over all B OD pairs), wall time, the final objective and
projected-gradient sup-norm, and the recovered (beta, gamma) vs theta_0.  Running
each basis answers the central question: BB is a first-order method whose rate scales
with cond(Q_B), so does the orthonormal reparametrization (which cuts the gamma-block
conditioning) make BB competitive with the affine-invariant Newton?

Run:
    python dev/bb_horserace.py
    python dev/bb_horserace.py --network SiouxFalls --th0 pos345 --B 200 --D 2000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from estim_dgp import CATALOG, build_network, make_ods, make_problem

from purc.estimators.debiased_fy import (
    DebiasedFYEstimator,
    DebiasedFYLoss,
    EstimatorConfig,
    GammaProjection,
    SieveBasis,
)
from purc.static_purc import SSNConfig
from purc.static_purc.dgp import simulate_dataset
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.utils.torch_compat import DEFAULT_DTYPE, to_numpy


def _solver():
    return IPMSolver(SSNConfig(max_iter=200), crossover=False, safeguard=True)


def _basis(name, L):
    return SieveBasis.monomial(L) if name == "monomial" else SieveBasis.orthonormal(L)


class CountingSolver:
    """Composition proxy that counts ``solve_batch`` calls (no monkey patching)."""

    def __init__(self, inner):
        self._inner = inner
        self.n_solve_batch = 0

    def solve_batch(self, *a, **k):
        self.n_solve_batch += 1
        return self._inner.solve_batch(*a, **k)

    def __getattr__(self, name):
        """Delegate any other attribute access to the wrapped solver."""
        return getattr(self._inner, name)


def spg_bb(
    loss,
    proj,
    layout,
    theta0,
    *,
    max_iter=800,
    tol=1e-7,
    mem=10,
    alpha_min=1e-12,
    alpha_max=1e12,
    sigma1=0.1,
    sigma2=0.9,
    gamma_ls=1e-4,
    max_ls=40,
):
    """
    Spectral Projected Gradient (projected BB1 + GLL nonmonotone line search).

    Minimizes ``loss`` over ``{theta : proj fixes the gamma block}`` from ``theta0``.
    Returns a result dict mirroring the estimator's fields.
    """

    def P(theta):
        beta, g = layout.unpack(theta)
        return layout.pack(beta, proj(g))

    theta = P(theta0.to(DEFAULT_DTYPE).reshape(-1))
    Q, grad = loss.value_and_grad(theta)
    fhist = [Q]
    # Spectral initial step: 1 / ||projected gradient||_inf.
    g0 = float((theta - P(theta - grad)).abs().max())
    if g0 == 0.0:
        return dict(theta=theta, n_outer=0, converged=True, Q=Q, gmap=0.0)
    alpha = min(alpha_max, max(alpha_min, 1.0 / g0))

    converged = False
    nit = 0
    for it in range(max_iter):
        nit = it + 1
        d = P(theta - alpha * grad) - theta  # feasible SPG direction
        delta = float(grad @ d)  # directional derivative (< 0 for a descent dir)
        if delta >= 0.0:  # numerically not descent: fall back to the largest step
            alpha = alpha_max
            d = P(theta - alpha * grad) - theta
            delta = float(grad @ d)
            if delta >= 0.0:
                converged = True  # projected gradient ~ 0
                break
        fmax = max(fhist[-mem:])
        lam = 1.0
        Qt = float("inf")
        for _ in range(max_ls):
            theta_t = theta + lam * d
            Qt = loss.value(theta_t)
            if np.isfinite(Qt) and Qt <= fmax + gamma_ls * lam * delta:
                break
            # safeguarded one-dimensional quadratic interpolation
            denom = Qt - Q - lam * delta
            lam_t = -0.5 * lam * lam * delta / denom if np.isfinite(denom) and denom > 0 else lam * 0.5
            lam = lam_t if (sigma1 * lam <= lam_t <= sigma2 * lam) else lam * 0.5
        theta_new = theta + lam * d
        Q_new, grad_new = loss.value_and_grad(theta_new)
        s = theta_new - theta
        y = grad_new - grad
        sy = float(s @ y)
        ss = float(s @ s)
        # BB1 step ss/sy, safeguarded; reset on non-positive curvature.
        alpha = alpha_max if sy <= 1e-16 * max(ss, 1e-30) else min(alpha_max, max(alpha_min, ss / sy))
        theta, Q, grad = theta_new, Q_new, grad_new
        fhist.append(Q)
        gmap = float((theta - P(theta - grad)).abs().max())
        if gmap < tol:
            converged = True
            break
    return dict(theta=theta, n_outer=nit, converged=converged, Q=Q, gmap=gmap)


def _cond_at(prob, data, th0, theta_mono, proj, basis_name):
    """cond(A) of the FD bread at a fixed theta, in the named basis (no fit -- robust)."""
    from purc.estimators.debiased_fy.variance import hessian_fd

    basis = _basis(basis_name, th0.L)
    loss = DebiasedFYLoss(prob, _preproc(prob), data, th0.L, warm_start=True, basis=basis)
    beta = as_t(theta_mono[: th0.beta.size])
    gamma_m = as_t(theta_mono[th0.beta.size :])
    c = gamma_m if basis.is_monomial else as_t(to_numpy(basis.from_monomial(gamma_m)))
    theta_c = loss.layout.pack(beta, c)
    A = hessian_fd(loss, theta_c)
    return float(torch.linalg.cond(A))


def as_t(x):
    """Numpy/torch -> float64 torch tensor."""
    return torch.as_tensor(np.asarray(x, float), dtype=DEFAULT_DTYPE)


def _preproc(prob):
    s = _solver()
    s.preprocess(prob)
    return s


def run_case(case):
    """Fit Newton and SPG on one verified instance; return a metrics dict."""
    th0 = CATALOG[case["th0"]]
    inc, attrs = build_network(case["net"], case["nseed"])
    prob = make_problem(inc, th0, seed=case["pseed"], attrs=attrs)
    n = inc.shape[0]
    ods = make_ods(n, case["B"], case["D"], case["odseed"])
    base = _preproc(prob)
    data = simulate_dataset(prob, base, (th0.beta, th0.gamma), ods, np.random.default_rng(case["simseed"]))
    proj_kind = case["proj"]
    P = th0.beta.size + (th0.L - 2)

    out = {"case": case["name"], "th0": case["th0"], "proj": proj_kind, "P": P}

    # --- Newton (current estimator), monomial basis, generous budget ---
    sN = CountingSolver(_solver())
    sN.preprocess(prob)
    cfg = EstimatorConfig(proj=GammaProjection(proj_kind), max_iter=200, tol_grad=1e-6)
    est = DebiasedFYEstimator(prob, sN, th0.L, cfg)
    t0 = time.perf_counter()
    rN = est.fit(data)
    out["newton"] = dict(
        outer=rN.n_outer, solves=sN.n_solve_batch, ms=(time.perf_counter() - t0) * 1e3,
        conv=rN.converged, Q=rN.objective, gmap=float(rN.grad.abs().max()),
        beta=to_numpy(rN.beta_hat).ravel(), gamma=to_numpy(rN.gamma_hat).ravel(),
        theta=to_numpy(rN.theta_hat).ravel(),
    )

    # --- SPG-BB on the identical loss/projection (monomial) ---
    sS = CountingSolver(_solver())
    sS.preprocess(prob)
    loss = DebiasedFYLoss(prob, sS, data, th0.L, warm_start=True)
    proj = GammaProjection(proj_kind)
    layout = loss.layout
    t0 = time.perf_counter()
    rS = spg_bb(loss, proj, layout, torch.zeros(layout.size, dtype=DEFAULT_DTYPE), max_iter=3000, tol=1e-6)
    beta_c, c = layout.unpack(rS["theta"])
    out["spg"] = dict(
        outer=rS["n_outer"], solves=sS.n_solve_batch, ms=(time.perf_counter() - t0) * 1e3,
        conv=rS["converged"], Q=rS["Q"], gmap=rS["gmap"],
        beta=to_numpy(beta_c).ravel(), gamma=to_numpy(c).ravel(), theta=to_numpy(rS["theta"]).ravel(),
    )
    out["th0_beta"] = np.asarray(th0.beta, float)
    out["th0_gamma"] = np.asarray(th0.gamma, float)
    out["agree"] = float(np.abs(out["newton"]["theta"] - out["spg"]["theta"]).max())
    # Conditioning the first-order method actually faces, monomial vs orthonormal.
    tn = out["newton"]["theta"]
    out["cond_mono"] = _cond_at(prob, data, th0, tn, proj_kind, "monomial")
    out["cond_orth"] = _cond_at(prob, data, th0, tn, proj_kind, "orthonormal") if P - th0.beta.size >= 2 else None
    return out


def _fmt(m, th0_b, th0_g):
    err_b = np.abs(m["beta"] - th0_b).max()
    err_g = np.abs(m["gamma"] - th0_g).max() if th0_g.size else 0.0
    return (
        f"{str(m['conv']):>5s} {m['outer']:>5d} {m['solves']:>7d} {m['ms']:>8.0f} "
        f"{m['Q']:>12.5e} {m['gmap']:>9.1e} {err_b:>8.1e} {err_g:>8.1e}"
    )


# Verified clean full-fit instances (see the seed sweep in the session notes):
#   well_cond : 1 sieve coef (P=3), bernstein -- well-conditioned control.
#   ill_cond  : 3 sieve coefs (P=5), nonneg (h''>=1, robust) -- the collinear
#               gamma-block (cond(A) ~ 1e4) that stresses a first-order method.
CASES = [
    dict(name="well_cond", th0="pos3", net="synth-20", nseed=2, pseed=3,
         B=80, D=200, odseed=7000, simseed=2000, proj="bernstein"),
    dict(name="ill_cond", th0="pos345", net="synth-30", nseed=0, pseed=1,
         B=400, D=1000, odseed=17000, simseed=5000, proj="nonneg"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single case by name")
    args = ap.parse_args()

    hdr = (f"  {'method':8s} {'conv':>5s} {'outer':>5s} {'solves':>7s} {'ms':>8s} "
           f"{'Q':>12s} {'|gmap|':>9s} {'errβ':>8s} {'errγ':>8s}")
    for case in CASES:
        if args.only and case["name"] != args.only:
            continue
        r = run_case(case)
        print(f"\n=== {case['name']} | theta0={case['th0']} ({case['net']}) | proj={case['proj']} "
              f"| B={case['B']} D={case['D']} | P={r['P']} ===")
        cond = f"cond(A) monomial={r['cond_mono']:.1e}"
        if r["cond_orth"] is not None:
            cond += f"  orthonormal={r['cond_orth']:.1e}  (basis cuts gamma-block conditioning)"
        print("  " + cond)
        print(hdr)
        print(f"  {'newton':8s} " + _fmt(r["newton"], r["th0_beta"], r["th0_gamma"]))
        print(f"  {'spg-bb':8s} " + _fmt(r["spg"], r["th0_beta"], r["th0_gamma"]))
        print(f"    -> newton/spg theta agree to {r['agree']:.1e}")


if __name__ == "__main__":
    main()
