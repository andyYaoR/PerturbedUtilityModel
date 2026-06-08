r"""
Comprehensive outer-solver benchmark & horse-race for the debiased-FY estimator.

The outer loop minimizes the **convex** loss ``Q_B(theta)``, ``theta = (beta in
R^K free, gamma in Gamma)``, with the inner forward solve a sieve
``h(xi) = xi^2/2 + sum_l gamma_l xi^l/l``.  The smoothness/corner regime the
optimizer faces is set by the projection (``nonneg`` -> ``h'' >= 1``, smooth
``x*``; ``bernstein`` -> ``m(gamma) -> 0`` near ``dGamma_B``, corner/kink regime)
and the ``gamma`` stiffness -- this is the "different perturbation" axis (the
estimator is sieve-locked).  We race, on the SAME loss/data/start per cell:

    newton        projected damped Newton (FD Hessian)
    tr_bfgs       trust-region damped-BFGS, native QP  (the new default)
    lbfgsb        scipy L-BFGS-B, closed-form grad     (nonneg box only)
    trust-constr  scipy TR quasi-Newton, BFGS model    (box or polyhedron)

All contenders are second-order / quasi-Newton; the first-order baselines (spectral
projected-gradient / Barzilai--Borwein, Nesterov universal fast gradient) were dropped
-- they need thousands of oracle calls on the ill-conditioned cells and lose decisively.
Cost unit is the number of inner batched IPM solves (``CountingSolver``), the real
expensive oracle; wall time is reported alongside.  Because ``Q_B`` is convex, the
unique minimizer is certified by the **agreement of heterogeneous solvers**; the
harness turns that into explicit gates:

    C1  analytic residual-form gradient  vs  central finite differences
    C2  cross-solver consensus on theta* (convex => unique min)
    C3  KKT/criticality + feasibility at each reported theta*
    C4  batched-IPM flows  vs  the independent scipy oracle at the consensus theta*

plus the already-established QP anchors (``tests/test_qp.py``: QP vs CVXPY, native
vs torch).  A separate QP micro-benchmark isolates the native kernel's value (the
outer loop is oracle-dominated, so the native QP only shows up timed directly).

Run:
    python dev/outer_horserace.py --tier smoke      # seconds (sanity)
    python dev/outer_horserace.py --tier quick      # synthetic grid, ~3-5 min
    python dev/outer_horserace.py --tier full --out results/outer_horserace.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from accel_horserace import lbfgsb
from bb_horserace import CountingSolver, _basis, _solver
from estim_dgp import CATALOG, build_network, make_ods, make_problem
from scipy.optimize import BFGS, Bounds, LinearConstraint, minimize

from purc.estimators.debiased_fy import (
    DebiasedFYEstimator,
    DebiasedFYLoss,
    EstimatorConfig,
    GammaProjection,
)
from purc.static_purc import PUMProblem, native_available
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.dgp import simulate_dataset
from purc.static_purc.oracle import solve_scipy
from purc.static_purc.perturbations._bernstein import bernstein_matrix
from purc.static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor, to_numpy

METHODS = ("newton", "tr_bfgs", "lbfgsb", "trust-constr")
TOL = 1e-6  # projected-gradient sup-norm target shared by all contenders

# Gate thresholds (see module docstring).  The convex certificate is Q-consensus:
# all converged solvers reach the SAME minimal objective.  theta disagreement is
# *reported* but not gated -- Q_B is convex yet NOT strictly convex (collinear sieve;
# CLAUDE.md), so the minimizer can be a flat valley and equal-Q theta's that differ in
# the weakly-identified gamma directions are legitimate, not a bug.
C1_TOL = 1e-6   # max absolute grad-vs-FD error (house style, h=1e-5; test_estimation.py)
C2_Q_TOL = 1e-7  # max objective spread among converged methods (the convex certificate)
C3_TOL = 1e-4   # max criticality (projected-gradient mapping) at a reported theta*
C4_TOL = 1e-6   # max |x_batch - x_scipy| at the consensus theta* (two iterative solvers)


# --------------------------------------------------------------------------- #
# scenario grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """One benchmark cell: a verified problem + a (projection, basis) regime."""

    name: str
    th0: str          # CATALOG key
    net: str          # "synth-<n>" or a TNTP name
    proj: str         # "nonneg" | "bernstein"
    basis: str        # "monomial" | "orthonormal"
    B: int
    D: int
    nseed: int = 2
    pseed: int = 3
    odseed: int = 7000
    simseed: int = 2000


def _cells(th0: str, net: str, B: int, D: int, *, bernstein_only: bool, **seeds) -> list[Scenario]:
    """The valid (proj, basis) regimes for one (th0, net, B, D), named uniformly."""
    regimes = [("bernstein", "monomial"), ("bernstein", "orthonormal")]
    if not bernstein_only:
        regimes = [("nonneg", "monomial"), *regimes]
    out = []
    for proj, basis in regimes:
        tag = f"{proj[:4]}/{basis[:4]}"
        out.append(Scenario(f"{th0}@{net}|{tag}", th0, net, proj, basis, B, D, **seeds))
    return out


def build_grid(tier: str) -> list[Scenario]:
    """The scenario list for a tier (smoke < quick < full)."""
    if tier == "smoke":
        return [
            Scenario("smoke/nonneg", "pos3", "synth-12", "nonneg", "monomial", 12, 80),
            Scenario("smoke/bernstein", "pos3", "synth-12", "bernstein", "monomial", 12, 80),
        ]
    grid: list[Scenario] = []
    # pos3 (P=3, well conditioned) and pos345 (P=5, collinear gamma block).
    grid += _cells("pos3", "synth-20", 80, 200, bernstein_only=False, nseed=2, pseed=3)
    grid += _cells("pos345", "synth-30", 80, 200, bernstein_only=False,
                   nseed=0, pseed=1, odseed=17000, simseed=5000)
    # neg (gamma has a negative coef) -> bernstein only.
    grid += _cells("neg", "synth-20", 80, 200, bernstein_only=True,
                   nseed=6, pseed=6, odseed=11000, simseed=4000)
    if tier == "full":
        # Larger sample on synthetic.
        grid += _cells("pos345", "synth-30", 400, 1000, bernstein_only=False,
                       nseed=0, pseed=1, odseed=18000, simseed=5500)
        # Real TNTP networks (SDDM path), small B, behind the degeneracy guard.
        for net in ("SiouxFalls", "ChicagoSketch"):
            grid += _cells("pos3", net, 60, 200, bernstein_only=False,
                           nseed=1, pseed=1, odseed=2200, simseed=3300)
    return grid


def applicable(method: str, proj: str, basis: str, th0: str) -> tuple[bool, str]:
    """Whether a contender can express this cell's constraint (else a reason string)."""
    if proj == "nonneg" and basis != "monomial":
        return False, "N/A (nonneg needs monomial)"
    if th0 == "neg" and proj == "nonneg":
        return False, "N/A (neg gamma not in R_+)"
    if method == "lbfgsb" and proj != "nonneg":
        return False, "N/A (needs polyhedron)"
    if method == "tr_bfgs-native" and not native_available():
        return False, "N/A (native absent)"
    return True, ""


# --------------------------------------------------------------------------- #
# problem / loss construction
# --------------------------------------------------------------------------- #
@dataclass
class Bundle:
    """A built scenario: problem, simulated data, basis, and layout facts."""

    prob: object
    th0: object
    basis: object
    data: object
    L: int
    n_beta: int
    P: int
    anchor_solver: object  # a preprocessed (non-counting) solver for the gates


def build_bundle(sc: Scenario) -> Optional[Bundle]:
    """Build one scenario; ``None`` if the forward solve degenerates on this draw."""
    th0 = CATALOG[sc.th0]
    try:
        inc, attrs = build_network(sc.net, sc.nseed)
        prob = make_problem(inc, th0, seed=sc.pseed, attrs=attrs)
        ods = make_ods(inc.shape[0], sc.B, sc.D, sc.odseed)
        base = _solver()
        base.preprocess(prob)
        data = simulate_dataset(prob, base, (th0.beta, th0.gamma), ods, np.random.default_rng(sc.simseed))
        anchor = _solver()
        anchor.preprocess(prob)
    except (np.linalg.LinAlgError, RuntimeError):
        return None
    basis = _basis(sc.basis, th0.L)
    n_beta = int(th0.beta.size)
    return Bundle(prob, th0, basis, data, th0.L, n_beta, n_beta + (th0.L - 2), anchor)


def _fresh_loss(bundle: Bundle):
    """A fresh basis-aware loss on a fresh ``CountingSolver`` (the per-method cost meter)."""
    s = CountingSolver(_solver())
    s.preprocess(bundle.prob)
    loss = DebiasedFYLoss(bundle.prob, s, bundle.data, bundle.L, warm_start=True, basis=bundle.basis)
    return loss, s


def _theta0_c(bundle: Bundle) -> torch.Tensor:
    """The ``c``-space start (zeros) -- feasible (gamma = 0 is the quadratic baseline)."""
    return torch.zeros(bundle.P, dtype=DEFAULT_DTYPE)


# --------------------------------------------------------------------------- #
# contenders (uniform result dict: theta_c, n_outer, converged, Q, solves, ms)
# --------------------------------------------------------------------------- #
def _trust_constr(loss, layout, theta0, n_beta, *, bern_M_c=None, tol=TOL, max_iter=400):
    """
    Scipy 'trust-constr' (TR quasi-Newton, BFGS model) on the box OR the polyhedron.

    No finite differences, no sensitivity, no line search -- a ratio test governs the
    radius and the BFGS model is built from gradient secants.  The nonneg box enters as
    simple bounds; the Bernstein polyhedron (in ``c``) enters as a ``LinearConstraint``.
    """
    cache: dict = {}

    def vg(x):
        key = x.tobytes()
        if key not in cache:
            Q, g = loss.value_and_grad(torch.as_tensor(x, dtype=DEFAULT_DTYPE))
            cache[key] = (float(Q), to_numpy(g).astype(np.float64))
        return cache[key]

    P = layout.size
    if bern_M_c is None:  # nonneg box
        lb = np.array([-np.inf] * n_beta + [0.0] * (P - n_beta))
        cons = []
    else:  # Bernstein polyhedron (M T) c >= -1 over the gamma block
        lb = np.full(P, -np.inf)
        A = np.hstack([np.zeros((bern_M_c.shape[0], n_beta)), bern_M_c])
        cons = [LinearConstraint(A, -1.0, np.inf)]
    res = minimize(
        lambda x: vg(x)[0], to_numpy(theta0).astype(np.float64), jac=lambda x: vg(x)[1],
        hess=BFGS(), method="trust-constr", bounds=Bounds(lb, np.full(P, np.inf)), constraints=cons,
        options={"maxiter": max_iter, "gtol": 1e-9, "xtol": 1e-14, "barrier_tol": 1e-10, "verbose": 0},
    )
    return dict(theta=torch.as_tensor(res.x, dtype=DEFAULT_DTYPE), n_outer=int(res.nit),
                converged=bool(res.status in (1, 2)), Q=float(res.fun))


def run_method(method: str, bundle: Bundle, sc: Scenario, caps: dict) -> dict:
    """Run one contender on a fresh loss; record (theta_c, n_outer, converged, Q, solves, ms)."""
    out = dict(method=method, theta_c=None, n_outer=0, converged=False,
               Q=float("nan"), solves=0, ms=0.0, error="")
    try:
        if method in ("newton", "tr_bfgs"):
            s = CountingSolver(_solver())
            s.preprocess(bundle.prob)
            cfg = EstimatorConfig(proj=GammaProjection(sc.proj), basis=bundle.basis,
                                  method=method, max_iter=caps["est"], tol_grad=TOL)
            est = DebiasedFYEstimator(bundle.prob, s, bundle.L, cfg)
            t0 = time.perf_counter()
            res = est.fit(bundle.data)
            out.update(theta_c=res.theta_hat.clone(), n_outer=res.n_outer, converged=bool(res.converged),
                       Q=float(res.objective), solves=s.n_solve_batch, ms=(time.perf_counter() - t0) * 1e3)
            return out

        loss, s = _fresh_loss(bundle)
        theta0 = _theta0_c(bundle)
        t0 = time.perf_counter()
        if method == "lbfgsb":
            r = lbfgsb(loss, loss.layout, theta0, bundle.n_beta, tol=TOL, max_iter=caps["fo"])
        elif method == "trust-constr":
            bM = None
            if sc.proj == "bernstein":
                bM = bundle.basis.constraint_matrix(bernstein_matrix(bundle.L - 2))
            r = _trust_constr(loss, loss.layout, theta0, bundle.n_beta, bern_M_c=bM, max_iter=caps["tc"])
        else:
            raise ValueError(f"unknown method {method!r}")
        out.update(theta_c=as_tensor(r["theta"]).to(DEFAULT_DTYPE), n_outer=int(r["n_outer"]),
                   converged=bool(r["converged"]), Q=float(r["Q"]), solves=s.n_solve_batch,
                   ms=(time.perf_counter() - t0) * 1e3)
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# --------------------------------------------------------------------------- #
# correctness gates
# --------------------------------------------------------------------------- #
def to_mono_theta(theta_c, bundle: Bundle) -> np.ndarray:
    """Map a ``c``-space ``theta`` to common monomial ``[beta, gamma]`` (consensus space)."""
    t = as_tensor(theta_c).to(DEFAULT_DTYPE).reshape(-1)
    beta = t[: bundle.n_beta]
    c = t[bundle.n_beta :]
    return to_numpy(torch.cat([beta, bundle.basis.to_monomial(c)]))


def criticality_and_feasible(gate_loss, theta_c, n_beta, proj_obj) -> tuple[float, bool]:
    """
    C3: ``||theta - P_Gamma(theta - g)||_inf`` (beta unconstrained) and gamma feasibility.

    Computed centrally with the *regime's* projection on a dedicated gate loss, so it
    does not pollute any contender's solve count.
    """
    theta = as_tensor(theta_c).to(DEFAULT_DTYPE).reshape(-1)
    _, g = gate_loss.value_and_grad(theta)
    step = theta - g
    c_step = step[n_beta:]
    proj_step = torch.cat([step[:n_beta], proj_obj(c_step)])  # beta unconstrained
    gmap = float(torch.max(torch.abs(theta - proj_step)))
    feasible = proj_obj.is_feasible(theta[n_beta:])
    return gmap, feasible


def gradient_audit(gate_loss, theta_c, *, h: float = 1e-5) -> float:
    """
    C1: max **absolute** error of the analytic gradient vs central FD (house style).

    The residual-form gradient is exact (it never differentiates through ``x*``); a
    coding bug would show an O(1) error, so a small absolute gap (floored by the
    iterative inner solve) confirms the formula.  Coordinates whose ``+/-h`` probe
    straddles ``dGamma_B`` (a non-finite value) are skipped.
    """
    theta = as_tensor(theta_c).to(DEFAULT_DTYPE).reshape(-1)
    _, g = gate_loss.value_and_grad(theta)
    g = to_numpy(g)
    worst = 0.0
    for j in range(theta.numel()):
        tp, tm = theta.clone(), theta.clone()
        tp[j] += h
        tm[j] -= h
        fp, fm = gate_loss.value(tp), gate_loss.value(tm)
        if not (np.isfinite(fp) and np.isfinite(fm)):
            continue  # a +/-h probe straddled dGamma_B -> skip this coordinate
        fd = (fp - fm) / (2.0 * h)
        worst = max(worst, abs(fd - g[j]))
    return worst


def consensus(results: dict, bundle: Bundle) -> dict:
    """
    C2: reference = converged method with the smallest criticality; max disagreement.

    Compared in monomial ``[beta, gamma]`` so methods on different bases are commensurable.
    """
    conv = {m: r for m, r in results.items() if r["converged"] and r["theta_c"] is not None}
    if not conv:
        return dict(ref=None, ref_method=None, disagree={}, max_disagree=float("nan"), q_spread=float("nan"))
    ref_m = min(conv, key=lambda m: (results[m]["gmap"], results[m]["Q"]))
    ref = to_mono_theta(conv[ref_m]["theta_c"], bundle)
    disagree = {m: float(np.abs(to_mono_theta(r["theta_c"], bundle) - ref).max()) for m, r in conv.items()}
    qs = [r["Q"] for r in conv.values()]
    return dict(ref=ref, ref_method=ref_m, disagree=disagree,
                max_disagree=max(disagree.values()), q_spread=float(max(qs) - min(qs)))


def inner_anchor(bundle: Bundle, ref_mono: np.ndarray, n_check: int = 6) -> float:
    """C4: batched-IPM flows vs the independent scipy oracle at the consensus theta*."""
    beta = ref_mono[: bundle.n_beta]
    gamma = ref_mono[bundle.n_beta :]
    b_batch = to_numpy(bundle.data.b_batch)
    xb = to_numpy(bundle.anchor_solver.solve_batch((beta, gamma), b_batch).x)
    A = bundle.prob.constraint.A
    ell = to_numpy(bundle.prob.constraint.ell)
    maxdiff = 0.0
    for i in range(min(len(b_batch), n_check)):
        prob_i = PUMProblem(bundle.prob.perturbation, GeneralPolytope(A, b_batch[i], ell=ell), Z=bundle.prob.Z)
        xo = solve_scipy(prob_i, (beta, gamma))
        maxdiff = max(maxdiff, float(np.max(np.abs(xb[i] - xo))))
    return maxdiff


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
_HDR = (f"  {'method':12s} {'conv':>5s} {'outer':>5s} {'solves':>7s} {'ms':>8s} "
        f"{'Q':>13s} {'|gmap|':>9s} {'feas':>5s} {'errβ':>8s} {'errγ':>8s} "
        f"{'disagr':>8s} {'×solv':>6s} {'×ms':>6s}")


def print_scenario(sc: Scenario, bundle: Bundle, results: dict, con: dict, c1: float, anchor: float) -> None:
    """Pretty per-scenario table with speedups vs newton and the gate read-outs."""
    base = results.get("newton")
    bsolv = base["solves"] if base and base["converged"] else None
    bms = base["ms"] if base and base["converged"] else None
    th0v = np.concatenate([bundle.th0.beta, bundle.th0.gamma])
    print(f"\n=== {sc.name} | {sc.net} | proj={sc.proj} basis={sc.basis} | B={sc.B} D={sc.D} | P={bundle.P} ===")
    print(f"  C1 grad-vs-FD={c1:.1e}  C2 Q-spread={con['q_spread']:.1e} "
          f"(ref={con['ref_method']})  θ-disagree={con['max_disagree']:.1e}  C4 oracle={anchor:.1e}")
    print(_HDR)
    for m in METHODS:
        r = results.get(m)
        if r is None:
            ok, why = applicable(m, sc.proj, sc.basis, sc.th0)
            print(f"  {m:12s} {why}")
            continue
        if r["theta_c"] is None:
            print(f"  {m:12s}  -- failed: {r['error']}")
            continue
        mono = to_mono_theta(r["theta_c"], bundle)
        eb = float(np.abs(mono[: bundle.n_beta] - bundle.th0.beta).max())
        eg = float(np.abs(mono[bundle.n_beta :] - bundle.th0.gamma).max()) if bundle.th0.gamma.size else 0.0
        dis = con["disagree"].get(m, float("nan"))
        sx = f"{bsolv / r['solves']:.1f}" if (bsolv and r["solves"]) else "-"
        mx = f"{bms / r['ms']:.1f}" if (bms and r["ms"]) else "-"
        print(f"  {m:12s} {str(r['converged']):>5s} {r['n_outer']:>5d} {r['solves']:>7d} {r['ms']:>8.0f} "
              f"{r['Q']:>13.6e} {r['gmap']:>9.1e} {str(r['feasible'])[:5]:>5s} {eb:>8.1e} {eg:>8.1e} "
              f"{dis:>8.1e} {sx:>6s} {mx:>6s}")


def qp_microbench(n_inst: int = 300, seed: int = 0) -> Optional[dict]:
    """Native vs torch ``solve_box_linear_qp`` over a random battery (parity + speedup)."""
    if not native_available():
        return None
    from purc.estimators.debiased_fy.qp import solve_box_linear_qp

    rng = np.random.default_rng(seed)
    insts = []
    for _ in range(n_inst):
        P = int(rng.integers(1, 9))
        Qm, _ = np.linalg.qr(rng.standard_normal((P, P)))
        eig = np.exp(np.linspace(0, np.log(10 ** rng.uniform(0, 5)), P))
        Bm = 0.5 * ((Qm * eig) @ Qm.T + (Qm * eig) @ Qm.T.T)
        g = rng.standard_normal(P) * rng.uniform(0.1, 8)
        d = rng.uniform(0.05, 3.0)
        m = int(rng.integers(0, 6))
        A = rng.standard_normal((m, P)) if m else None
        a = -np.abs(rng.standard_normal(m)) if m else None
        insts.append((Bm, g, -d * np.ones(P), d * np.ones(P), A, a))

    def run(prefer):
        t0 = time.perf_counter()
        ds = [solve_box_linear_qp(B, g, lo, hi, A=A, a=a, prefer_native=prefer).d.numpy()
              for (B, g, lo, hi, A, a) in insts]
        return (time.perf_counter() - t0), ds

    run(True)  # warm the native dispatch
    t_t, d_t = run(False)
    t_n, d_n = run(True)
    parity = max(float(np.abs(a - b).max()) for a, b in zip(d_t, d_n))
    return dict(n=n_inst, torch_ms=t_t * 1e3, native_ms=t_n * 1e3,
                speedup=t_t / t_n, max_parity_diff=parity)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
_CAPS = {
    "smoke": dict(est=60, fo=800, tc=200),
    "quick": dict(est=200, fo=3000, tc=400),
    "full": dict(est=300, fo=4000, tc=500),
}


def run_scenario(sc: Scenario, caps: dict, methods: tuple) -> Optional[dict]:
    """Build, race the applicable methods, run the gates; return a JSON-able summary."""
    bundle = build_bundle(sc)
    if bundle is None:
        print(f"\n=== {sc.name} | {sc.net} ===  SKIPPED (forward solve degenerated on this draw)")
        return dict(name=sc.name, skipped=True)
    gate_loss, _ = _fresh_loss(bundle)
    proj_obj = GammaProjection(sc.proj, basis=bundle.basis)

    results: dict = {}
    for m in methods:
        ok, _ = applicable(m, sc.proj, sc.basis, sc.th0)
        if not ok:
            continue
        results[m] = run_method(m, bundle, sc, caps)
        if results[m]["theta_c"] is not None:
            gmap, feas = criticality_and_feasible(gate_loss, results[m]["theta_c"], bundle.n_beta, proj_obj)
            results[m]["gmap"], results[m]["feasible"] = gmap, feas

    # C1 at two interior, nonzero-beta points (beta=0 makes utilities vanish -> corner
    # flows where the FD is noisy; the analytic gradient is exact regardless).
    def _interior(scale_b, scale_g):
        return torch.cat([as_tensor(scale_b * bundle.th0.beta).to(DEFAULT_DTYPE),
                          scale_g * bundle.basis.from_monomial(as_tensor(bundle.th0.gamma)).to(DEFAULT_DTYPE)])

    c1 = max(gradient_audit(gate_loss, _interior(1.0, 1.0)), gradient_audit(gate_loss, _interior(0.8, 0.5)))

    con = consensus(results, bundle)
    anchor = inner_anchor(bundle, con["ref"]) if con["ref"] is not None else float("nan")
    print_scenario(sc, bundle, results, con, c1, anchor)

    feas_all = all(r.get("feasible", True) for r in results.values() if r["theta_c"] is not None)
    gmap_worst = max((r["gmap"] for r in results.values()
                      if r["converged"] and r["theta_c"] is not None), default=float("nan"))
    rows = {m: {k: v for k, v in r.items() if k != "theta_c"} for m, r in results.items()}
    return dict(name=sc.name, net=sc.net, proj=sc.proj, basis=sc.basis, P=bundle.P,
                c1=c1, c2_q_spread=con["q_spread"], c2_max_disagree=con["max_disagree"],
                c2_ref=con["ref_method"], c3_worst_gmap=gmap_worst, c3_all_feasible=feas_all,
                c4_oracle=anchor, rows=rows)


def rollup(summaries: list[dict]) -> dict:
    """Aggregate gates + win-rates across scenarios; print a PASS/FAIL banner."""
    live = [s for s in summaries if not s.get("skipped")]
    wins = {m: 0 for m in METHODS}
    nonconv = {m: 0 for m in METHODS}
    speedups: dict = {m: [] for m in METHODS}
    for s in live:
        rows = s["rows"]
        conv = {m: r for m, r in rows.items() if r["converged"]}
        if conv:
            win = min(conv, key=lambda m: conv[m]["solves"] or 1 << 30)
            wins[win] += 1
        base = rows.get("newton")
        for m, r in rows.items():
            if not r["converged"]:
                nonconv[m] += 1
            if base and base["converged"] and r["converged"] and r["solves"]:
                speedups[m].append(base["solves"] / r["solves"])
    worst_c1 = max((s["c1"] for s in live), default=float("nan"))
    worst_c2 = max((s["c2_q_spread"] for s in live
                    if np.isfinite(s["c2_q_spread"])), default=float("nan"))
    worst_theta = max((s["c2_max_disagree"] for s in live
                       if np.isfinite(s["c2_max_disagree"])), default=float("nan"))
    worst_c3 = max((s["c3_worst_gmap"] for s in live
                    if np.isfinite(s["c3_worst_gmap"])), default=float("nan"))
    feas_ok = all(s["c3_all_feasible"] for s in live)
    worst_c4 = max((s["c4_oracle"] for s in live if np.isfinite(s["c4_oracle"])), default=float("nan"))

    print("\n" + "=" * 78)
    print("ROLL-UP")
    print("=" * 78)
    print(f"  scenarios: {len(live)} run, {len(summaries) - len(live)} skipped (degenerate)")
    print(f"  {'method':12s} {'wins':>5s} {'non-conv':>8s} {'median ×solves vs newton':>26s}")
    for m in METHODS:
        med = float(np.median(speedups[m])) if speedups[m] else float("nan")
        print(f"  {m:12s} {wins[m]:>5d} {nonconv[m]:>8d} {med:>26.2f}")
    gates = {
        "C1 grad-vs-FD": (worst_c1, worst_c1 < C1_TOL),
        "C2 Q-consensus": (worst_c2, not np.isfinite(worst_c2) or worst_c2 < C2_Q_TOL),
        "C3 criticality": (worst_c3, (not np.isfinite(worst_c3) or worst_c3 < C3_TOL) and feas_ok),
        "C4 oracle": (worst_c4, not np.isfinite(worst_c4) or worst_c4 < C4_TOL),
    }
    print()
    for name, (val, ok) in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} worst={val:.2e}")
    print(f"  [info] θ-disagree (flat-direction drift, not a fault) worst={worst_theta:.2e}")
    return dict(wins=wins, nonconv=nonconv, worst_theta_disagree=float(worst_theta),
                median_speedup={m: (float(np.median(speedups[m])) if speedups[m] else None) for m in METHODS},
                gates={k: {"worst": float(v[0]), "pass": bool(v[1])} for k, v in gates.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="quick", choices=["smoke", "quick", "full"])
    ap.add_argument("--only", default=None, help="run a single scenario by name")
    ap.add_argument("--out", default=None, help="write the JSON summary to this path")
    ap.add_argument("--no-qp-bench", action="store_true", help="skip the QP micro-benchmark")
    args = ap.parse_args()

    caps = _CAPS[args.tier]
    grid = [sc for sc in build_grid(args.tier) if not args.only or sc.name == args.only]
    summaries = [run_scenario(sc, caps, METHODS) for sc in grid]
    roll = rollup(summaries)

    qp = None if args.no_qp_bench else qp_microbench()
    if qp:
        print(f"\n  QP kernel (native vs torch, {qp['n']} solves): "
              f"speedup ×{qp['speedup']:.1f}  parity {qp['max_parity_diff']:.1e}  "
              f"(torch {qp['torch_ms']:.0f}ms, native {qp['native_ms']:.0f}ms)")

    if args.out:
        from pathlib import Path
        blob = json.dumps(dict(tier=args.tier, scenarios=summaries, rollup=roll, qp_microbench=qp),
                          indent=2, default=float)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(blob)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
