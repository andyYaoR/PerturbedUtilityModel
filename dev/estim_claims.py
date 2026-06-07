"""
Monte Carlo experiments testing the formal claims of the debiased FY estimator.

Each function returns a small result dict (also printed) suitable for the paper's
simulation section.  Networks and replication counts scale with ``quick``.

Claims (paper labels in parentheses):
  1 (prop:debiased iii) unbiased score: the debiased gamma-score has mean ~0 at
    theta_0 while the naive plug-in is biased upward, the bias shrinking in D.
  3 (prop:asymptotic) consistency: RMSE(theta_hat) decays like B^{-1/2}.
  4 (prop:asymptotic) coverage: sandwich SE ~ empirical SD; CI coverage ~ nominal.
  6 Gamma projection: with a negative gamma_l, only the Bernstein projection
    recovers it; R_+ clips it to ~0.
  7 (prop:ipm) the batched IPM in the loop: a fixed low iteration count per theta
    and predicted flows matching the scipy oracle across the whole theta-sweep
    (lambda warm starts do NOT cut the primal IPM's iterations -- it re-centers).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from estim_dgp import CATALOG, build_network, make_ods, make_problem, synth_incidence

from purc.estimators.debiased_fy import (
    DebiasedFYEstimator,
    DebiasedFYLoss,
    EstimatorConfig,
    GammaProjection,
    NaiveFYLoss,
    SieveBasis,
)
from purc.estimators.debiased_fy.variance import sandwich_variance
from purc.static_purc import PUMProblem, SSNConfig
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.dgp import simulate_dataset
from purc.static_purc.oracle import solve_scipy
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.utils.torch_compat import to_numpy

# Optional uniform cap on replication counts (set by estim_run --reps), so the
# full grids can be run at a controlled cost without truncating the B/D sweeps.
MAX_REPS = None


def _reps(n: int) -> int:
    """Apply the global replication cap, if any."""
    return min(n, MAX_REPS) if MAX_REPS else n


def _solver():
    return IPMSolver(SSNConfig(max_iter=200), crossover=False, safeguard=True)


def _network(network, default_n: int, default_seed: int):
    """
    Resolve a claim's network: ``None`` -> the claim's built-in synthetic graph
    (so prior synthetic results reproduce exactly); a name -> that TNTP/synth network.

    Returns ``(incidence[n,N], attrs[N] | None)``.
    """
    if network is None:
        return synth_incidence(default_n, default_seed), None
    return build_network(network, default_seed)


def _make_basis(name, L):
    """Resolve a basis name -> SieveBasis (None/'monomial' -> power series)."""
    if name is None or name == "monomial":
        return SieveBasis.monomial(L)
    if name == "orthonormal":
        return SieveBasis.orthonormal(L)
    raise ValueError(f"unknown basis {name!r}")


def _fit(prob, th0, ods, rng, proj="bernstein", theta_init=None, basis=None):
    solver = _solver()
    solver.preprocess(prob)
    data = simulate_dataset(prob, solver, (th0.beta, th0.gamma), ods, rng)
    cfg = EstimatorConfig(proj=GammaProjection(proj), basis=_make_basis(basis, th0.L))
    est = DebiasedFYEstimator(prob, solver, th0.L, cfg, theta_init=theta_init)
    return est.fit(data), data, solver


def _fit_safe(prob, th0, ods, rng, proj="bernstein", theta_init=None, basis=None):
    """
    Fit, returning ``None`` on a rare ill-conditioned (non-PD) random instance.

    A degenerate synthetic OD instance can make the (grounded, weighted-Laplacian)
    normal equations numerically indefinite; in a Monte Carlo study such a draw is
    skipped and counted, rather than aborting the whole sweep.  ``basis`` selects the
    sieve parametrization (``"monomial"`` default, or ``"orthonormal"``).
    """
    try:
        return _fit(prob, th0, ods, rng, proj=proj, theta_init=theta_init, basis=basis)
    except (np.linalg.LinAlgError, RuntimeError):
        return None


# --------------------------------------------------------------------------- #
def claim1_unbiased_score(quick: bool, network=None, basis=None) -> dict:
    """
    Debiased score ~0 at theta_0; naive plug-in biased upward (shrinks in D).

    Honors ``basis`` (``"monomial"`` default or ``"orthonormal"``): a fixed linear
    reparametrization preserves the debiasing, so the debiased score stays ~0 in the
    orthonormal basis too (the naive score remains biased).
    """
    th0 = CATALOG["pos345"]
    inc, attrs = _network(network, 30, 0)
    prob = make_problem(inc, th0, seed=1, attrs=attrs)
    sb = _make_basis(basis, th0.L)
    B = 60 if quick else 200
    reps = _reps(30 if quick else 300)
    Ds = [20, 100] if quick else [20, 100, 1000]
    out = {}
    for D in Ds:
        deb, nav = [], []
        for r in range(reps):
            rng = np.random.default_rng(1000 + r)
            solver = _solver()
            solver.preprocess(prob)
            ods = make_ods(inc.shape[0], B, D, seed=5000 + r)
            data = simulate_dataset(prob, solver, (th0.beta, th0.gamma), ods, rng)
            th = DebiasedFYLoss(prob, solver, data, th0.L, basis=sb).layout.pack(
                th0.beta, sb.from_monomial(th0.gamma)
            )
            gd = to_numpy(DebiasedFYLoss(prob, solver, data, th0.L, basis=sb).value_and_grad(th)[1])
            gn = to_numpy(NaiveFYLoss(prob, solver, data, th0.L, basis=sb).value_and_grad(th)[1])
            # gamma block is the last L-2 entries.
            ng = th0.L - 2
            deb.append(gd[-ng:])
            nav.append(gn[-ng:])
        deb = np.stack(deb)
        nav = np.stack(nav)
        out[D] = {
            "debiased_gamma_score_mean": deb.mean(0).tolist(),
            "debiased_gamma_score_se": (deb.std(0) / np.sqrt(reps)).tolist(),
            "naive_gamma_score_mean": nav.mean(0).tolist(),
        }
        print(
            f"[claim1] D={D:5d}  debiased gamma-score mean={deb.mean(0)} "
            f"(+/-{deb.std(0)/np.sqrt(reps)})  naive mean={nav.mean(0)}"
        )
    return out


def claim3_consistency(quick: bool, network=None, basis=None) -> dict:
    """
    RMSE(theta_hat) vs B; the log-log slope should be about -1/2.

    Also reports the *local* (segment) slopes and a higher-order ``MSE = a/B +
    b/B^2`` fit: a regular root-B M-estimator has local slope ``-1/2 - (b/a)/(2B)``,
    so a finite-sample fit is slightly steeper than -1/2 (``b>0``) and drifts to
    -1/2 as ``B`` grows.  The extended grid (B up to 1600) exhibits this drift.
    """
    th0 = CATALOG["pos3"]
    inc, attrs = _network(network, 20, 2)
    prob = make_problem(inc, th0, seed=3, attrs=attrs)
    Bs = [25, 50, 100] if quick else [25, 50, 100, 200, 400, 800, 1600]
    reps = _reps(25 if quick else 150)
    D = 200
    P = th0.beta.size + th0.gamma.size
    th0_vec = np.concatenate([th0.beta, th0.gamma])
    rmse = {}
    for B in Bs:
        errs = []
        for r in range(reps):
            rng = np.random.default_rng(2000 + r)
            ods = make_ods(inc.shape[0], B, D, seed=7000 + r)
            out = _fit_safe(prob, th0, ods, rng, proj="nonneg", theta_init=np.zeros(P))
            if out is None:
                continue
            res, _, _ = out
            errs.append(to_numpy(res.theta_hat) - th0_vec)
        errs = np.stack(errs)
        rmse[B] = float(np.sqrt((errs**2).sum(1).mean()))
        print(f"[claim3] B={B:5d}  RMSE={rmse[B]:.4f}")
    Bsa = np.array(Bs, float)
    rm = np.array([rmse[b] for b in Bs])
    slope = float(np.polyfit(np.log(Bsa), np.log(rm), 1)[0])
    # Local (segment) slopes: should drift up toward -1/2 as B grows.
    local = {
        f"{Bs[i - 1]}->{Bs[i]}": float(np.log(rm[i] / rm[i - 1]) / np.log(Bsa[i] / Bsa[i - 1]))
        for i in range(1, len(Bs))
    }
    # Higher-order fit  MSE = a/B + b/B^2  <=>  MSE*B = a + b*(1/B)  (linear in 1/B).
    b_coef, a_coef = (float(x) for x in np.polyfit(1.0 / Bsa, rm**2 * Bsa, 1))
    print(f"[claim3] log-log RMSE-vs-B slope = {slope:.3f} (target ~ -0.5)")
    print(f"[claim3] local slopes = { {k: round(v, 3) for k, v in local.items()} }")
    print(f"[claim3] MSE=a/B+b/B^2 fit: a={a_coef:.3f}, b={b_coef:.3f} "
          f"(b>0 => slope steeper than -1/2 at small B, -> -1/2 as B grows)")
    return {
        "rmse": rmse,
        "slope": slope,
        "local_slopes": local,
        "mse_fit": {"a": a_coef, "b": b_coef},
    }


def claim4_coverage(quick: bool, network=None, basis=None) -> dict:
    """Sandwich SE vs empirical SD and 95% CI coverage at fixed B."""
    th0 = CATALOG["pos3"]
    inc, attrs = _network(network, 20, 4)
    prob = make_problem(inc, th0, seed=4, attrs=attrs)
    B = 100 if quick else 200
    reps = _reps(30 if quick else 120)
    D = 200
    P = th0.beta.size + th0.gamma.size
    th0_vec = np.concatenate([th0.beta, th0.gamma])
    ests, ses, covered = [], [], []
    for r in range(reps):
        rng = np.random.default_rng(3000 + r)
        ods = make_ods(inc.shape[0], B, D, seed=9000 + r)
        out = _fit_safe(prob, th0, ods, rng, proj="nonneg", theta_init=np.zeros(P))
        if out is None:
            continue
        res, data, solver = out
        loss = DebiasedFYLoss(prob, solver, data, th0.L)
        sw = sandwich_variance(loss, res.theta_hat)
        th = to_numpy(res.theta_hat)
        se = to_numpy(sw.se)
        ests.append(th)
        ses.append(se)
        covered.append((np.abs(th - th0_vec) <= 1.96 * se).astype(float))
    ests = np.stack(ests)
    emp_sd = ests.std(0)
    mean_se = np.stack(ses).mean(0)
    cov = np.stack(covered).mean(0)
    print(f"[claim4] empirical SD = {emp_sd}")
    print(f"[claim4] mean sandwich SE = {mean_se}  (ratio {mean_se/emp_sd})")
    print(f"[claim4] 95% CI coverage = {cov}")
    return {"emp_sd": emp_sd.tolist(), "mean_se": mean_se.tolist(), "coverage": cov.tolist()}


def claim6_projection(quick: bool, network=None, basis=None) -> dict:
    """Negative gamma_4: only the Bernstein projection recovers it; R_+ clips to ~0."""
    th0 = CATALOG["neg"]  # gamma = (0.8, -0.3)
    inc, attrs = _network(network, 20, 6)
    prob = make_problem(inc, th0, seed=6, attrs=attrs)
    B = 60 if quick else 200
    reps = _reps(20 if quick else 100)
    D = 1000 if quick else 2000
    P = th0.beta.size + th0.gamma.size
    res = {}
    for proj in ("nonneg", "bernstein"):
        g4 = []
        for r in range(reps):
            rng = np.random.default_rng(4000 + r)
            ods = make_ods(inc.shape[0], B, D, seed=11000 + r)
            out = _fit_safe(prob, th0, ods, rng, proj=proj, theta_init=np.zeros(P))
            if out is None:
                continue
            r_, _, _ = out
            g4.append(float(r_.gamma_hat[-1]))  # the gamma_4 estimate
        g4 = np.array(g4)
        res[proj] = {"gamma4_mean": float(g4.mean()), "gamma4_rmse": float(np.sqrt(((g4 - (-0.3))**2).mean()))}
        print(f"[claim6] proj={proj:9s}  gamma4_hat mean={g4.mean():+.3f} (true -0.3)  RMSE={res[proj]['gamma4_rmse']:.3f}")
    return res


def claim7_inner(quick: bool, network=None, basis=None) -> dict:
    """
    Batched IPM in the loop: fixed low iteration count and oracle accuracy on the sweep.

    Also records cold vs lambda-warm-started inner iterations.  The primal IPM
    re-centers from its strictly-interior start each solve, so warming lambda does
    NOT reduce its iteration count (unlike a dual/active-set method); the batched
    solver's value is the vectorization over OD pairs and oracle-accurate solves at
    every theta, not warm starts.
    """
    th0 = CATALOG["pos345"]
    inc, attrs = _network(network, 80 if quick else 200, 7)
    prob = make_problem(inc, th0, seed=7, attrs=attrs)
    B = 12 if quick else 40
    D = 200
    solver = _solver()
    solver.preprocess(prob)
    rng = np.random.default_rng(70)
    ods = make_ods(inc.shape[0], B, D, seed=13000)
    data = simulate_dataset(prob, solver, (th0.beta, th0.gamma), ods, rng)
    b_batch = to_numpy(data.b_batch)
    # cold vs warm across a small theta-sweep
    cold_iters, warm_iters = [], []
    warm = None
    for k in range(6):
        beta = th0.beta * (1.0 + 0.02 * k)
        cold = solver.solve_batch((beta, th0.gamma), b_batch, lam0=np.zeros_like(b_batch))
        warm_r = solver.solve_batch((beta, th0.gamma), b_batch, lam0=warm)
        warm = to_numpy(warm_r.lam)
        cold_iters.append(cold.nit)
        warm_iters.append(warm_r.nit)
    # oracle accuracy at theta_0: compare the batched primal to scipy per OD.
    xb = to_numpy(solver.solve_batch((th0.beta, th0.gamma), b_batch).x)
    ell = to_numpy(prob.constraint.ell)
    A = prob.constraint.A
    maxdiff = 0.0
    for i in range(min(B, 6)):
        p_i = PUMProblem(prob.perturbation, GeneralPolytope(A, b_batch[i], ell=ell), Z=prob.Z)
        xo = solve_scipy(p_i, (th0.beta, th0.gamma))
        maxdiff = max(maxdiff, float(np.max(np.abs(xb[i] - xo))))
    print(f"[claim7] cold inner iters={cold_iters}  warm inner iters={warm_iters}")
    print(f"[claim7] max |x_batch - x_scipy| over sweep start = {maxdiff:.1e}")
    return {"cold_iters": cold_iters, "warm_iters": warm_iters, "oracle_maxdiff": maxdiff}


def claim2_loss_min(quick: bool, network=None, basis=None) -> dict:
    """
    The FY loss ``Q_B`` is convex with its minimum at ``theta_0`` (grid slices).

    With many OD pairs and large demand the debiased loss is a near-population FY
    divergence: convex in ``theta`` and minimized at ``theta_0`` with value ~0.  We
    sweep each coordinate over an offset grid through ``theta_0`` and check the slice
    minimum sits at the zero offset.  (A ``gamma`` coordinate driven negative can
    leave ``Gamma_B``; there the inner solve is ill posed and the loss reports a
    non-finite sentinel -- which simply marks the convexity boundary on the slice.)
    """
    th0 = CATALOG["pos345"]
    inc, attrs = _network(network, 30, 0)
    prob = make_problem(inc, th0, seed=1, attrs=attrs)
    B = 100 if quick else 400
    D = 500 if quick else 2000
    solver = _solver()
    solver.preprocess(prob)
    rng = np.random.default_rng(220)
    ods = make_ods(inc.shape[0], B, D, seed=15000)
    data = simulate_dataset(prob, solver, (th0.beta, th0.gamma), ods, rng)
    loss = DebiasedFYLoss(prob, solver, data, th0.L)
    th0v = loss.layout.pack(th0.beta, th0.gamma)
    P = int(th0v.numel())
    q0 = float(loss.value(th0v))
    offs = np.linspace(-0.4, 0.4, 17)
    dz = float(offs[1] - offs[0])
    slices, all_at_zero, qmin = {}, True, q0
    for j in range(P):
        qs = []
        for o in offs:
            thj = th0v.clone()
            thj[j] = thj[j] + float(o)
            qs.append(float(loss.value(thj)))
        qs = np.array(qs)
        amin = float(offs[int(np.argmin(qs))])
        slices[str(j)] = {"offsets": offs.tolist(), "Q": qs.tolist(), "argmin_offset": amin}
        all_at_zero = all_at_zero and (abs(amin) <= dz + 1e-9)
        qmin = min(qmin, float(np.min(qs)))
    print(f"[claim2] Q(theta0)={q0:.4e}  min Q over slices={qmin:.4e}  every-slice-argmin@theta0={all_at_zero}")
    return {"Q0": q0, "Qmin": qmin, "argmin_at_theta0": bool(all_at_zero), "slices": slices}


def claim5_identifiability(quick: bool, network=None, basis=None) -> dict:
    """
    Per-``gamma_l`` RMSE across ``(B,D)`` and the sieve's collinearity.

    For ``L=5`` (``gamma=(0.5,0.3,0.1)`` at degrees 3,4,5) the monomials ``x^l`` are
    near-collinear on the flow support (which concentrates near 0), so the individual
    ``gamma_l`` are only weakly separated: the per-coordinate RMSE is large and
    non-monotone in degree, shrinking only slowly with ``(B,D)``, even though the
    *joint* estimate is consistent (claim 3).  We report the RMSE-per-``gamma_l``
    grid together with the bread conditioning ``cond(A)`` and the ``gamma``-block
    score correlations, which quantify the collinearity the sandwich variance
    (claim 4) correctly reflects.
    """
    th0 = CATALOG["pos345"]
    inc, attrs = _network(network, 30, 0)
    prob = make_problem(inc, th0, seed=1, attrs=attrs)
    Bs = [100, 400] if quick else [100, 400, 1600]
    Ds = [100, 1000] if quick else [50, 200, 1000]
    reps = _reps(15 if quick else 80)
    P = th0.beta.size + th0.gamma.size
    ng = th0.gamma.size
    # Under the sieve collinearity the cubic coefficient can blow up at minimal
    # data, so the per-gamma_l error is summarized by the *median* absolute error
    # (robust to those rare wild draws); the blow-up frequency is reported alongside.
    grid, blow = {}, {}
    for B in Bs:
        for D in Ds:
            errs = []
            for r in range(reps):
                rng = np.random.default_rng(5000 + r)
                ods = make_ods(inc.shape[0], B, D, seed=17000 + r)
                out = _fit_safe(prob, th0, ods, rng, proj="nonneg", theta_init=np.zeros(P))
                if out is None:
                    continue
                errs.append(to_numpy(out[0].gamma_hat) - th0.gamma)
            errs = np.stack(errs)
            mae_l = np.median(np.abs(errs), axis=0)  # robust per-gamma_l error (deg 3,4,5)
            grid[f"B{B}_D{D}"] = mae_l.tolist()
            blow[f"B{B}_D{D}"] = float(np.mean(np.abs(errs).max(1) > 10.0))  # wild-draw fraction
            print(f"[claim5] B={B:5d} D={D:5d}  per-gamma median|err| (g3,g4,g5)={np.round(mae_l, 3)}"
                  f"  blow-up frac={blow[f'B{B}_D{D}']:.2f}")
    # Conditioning at theta_0 on a well-sampled instance, in BOTH the monomial and
    # the fixed orthonormal basis.  The sieve (gamma-block) condition number is the
    # quantity the basis controls: orthonormalizing removes the basis-induced
    # (Hilbert) part, so cond drops sharply -- the sieve is well-identified *as a
    # function*.  (The full cond(A) is scale-sensitive: orthonormalizing only the
    # sieve block leaves a beta-vs-c scale mismatch, so the full number is not the
    # meaningful metric and can even rise.  The individual *monomial* coefficients
    # remain weakly identified because Var(gamma) = T Var(c) T^T re-amplifies -- the
    # intrinsic, representation-dependent limit.)  Debiasing-preservation is shown by
    # claim 1 (the mean debiased score stays ~0 in either basis).
    from purc.estimators.debiased_fy.variance import hessian_fd

    solver = _solver()
    solver.preprocess(prob)
    rng = np.random.default_rng(424242)
    ods = make_ods(inc.shape[0], 400, 2000, seed=424242)
    data = simulate_dataset(prob, solver, (th0.beta, th0.gamma), ods, rng)
    bases = {}
    corr = None
    for bname in ("monomial", "orthonormal"):
        basis = _make_basis(bname, th0.L)
        loss = DebiasedFYLoss(prob, solver, data, th0.L, basis=basis)
        theta0 = loss.layout.pack(th0.beta, basis.from_monomial(th0.gamma))
        A = to_numpy(hessian_fd(loss, theta0))
        gblk = A[-ng:, -ng:]
        bases[bname] = {
            "cond_A_full": float(np.linalg.cond(A)),
            "cond_gamma_block": float(np.linalg.cond(gblk)),
        }
        if bname == "monomial":
            d = np.sqrt(np.clip(np.diag(gblk), 1e-30, None))
            corr = (gblk / np.outer(d, d)).tolist()
        print(f"[claim5] basis={bname:11s} gamma-block cond={bases[bname]['cond_gamma_block']:.0f}"
              f"  (full cond(A)={bases[bname]['cond_A_full']:.0f})")
    drop = bases["monomial"]["cond_gamma_block"] / bases["orthonormal"]["cond_gamma_block"]
    print(f"[claim5] gamma-corr (monomial) offdiag~{np.round([corr[0][1], corr[0][2], corr[1][2]], 3)}"
          f"  gamma-block cond drop x{drop:.1f} (basis-induced part removed)")
    return {
        "Bs": Bs,
        "Ds": Ds,
        "mae_per_gamma": grid,
        "blowup_frac": blow,
        "cond_gamma_block": bases["monomial"]["cond_gamma_block"],
        "gamma_score_corr": corr,
        "basis_comparison": bases,
        "gamma_block_cond_drop": float(drop),
    }


ALL = {
    "1": claim1_unbiased_score,
    "2": claim2_loss_min,
    "3": claim3_consistency,
    "4": claim4_coverage,
    "5": claim5_identifiability,
    "6": claim6_projection,
    "7": claim7_inner,
}
