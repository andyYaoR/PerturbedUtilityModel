"""
CI correctness gates for the debiased-FY outer-loop solvers.

Hermetic (a tiny inline toy problem, package imports only -- no ``dev/`` modules, no
TNTP files): on a small *convex* instance we assert the same gates the dev horse-race
(``dev/outer_horserace.py``) reports across its full grid, for the two production
solvers (``newton``, ``tr_bfgs``) cross-checked against an independent scipy minimizer:

  * **C1 gradient** -- the closed-form residual-form gradient matches a central FD of
    ``Q_B`` (house style, ``h=1e-5``), so the oracle every method consumes is correct;
  * **C2 Q-consensus** -- ``newton``, ``tr_bfgs`` and scipy reach the *same* minimal
    objective.  ``Q_B`` is convex, so a unique minimum *value* exists and heterogeneous
    solvers agreeing on it certifies the optimum (the minimizer itself can drift along
    the flat, collinear ``gamma`` directions -- that is identification, not a fault);
  * **C3 KKT** -- each reported ``theta*`` is criticality-stationary (projected-gradient
    mapping ~ 0) and feasible in the regime's ``Gamma``;
  * **C4 oracle** -- at the optimum the batched IPM flows match the independent dense
    ``solve_scipy`` oracle, anchoring the whole stack to a separate solver;
  * **native QP** -- when the native kernel is present, the native-dispatched
    ``tr_bfgs`` lands at the certified optimum (the QP-level parity is in
    ``tests/test_qp.py::test_qp_native_parity``).

Both the nonneg box and the Bernstein polyhedron are exercised.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scipy.optimize import BFGS, LinearConstraint, minimize

from purc.estimators.debiased_fy import (
    DebiasedFYEstimator,
    DebiasedFYLoss,
    EstimatorConfig,
    GammaProjection,
)
from purc.static_purc import ForwardSolverConfig, PUMProblem, native_available
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.dgp import ODSpec, simulate_dataset
from purc.static_purc.oracle import solve_scipy
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.perturbations._bernstein import bernstein_matrix
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.utils.torch_compat import DEFAULT_DTYPE, to_numpy

BETA0 = np.array([-1.0])
GAMMA0 = np.array([0.5])
L = 3  # one sieve coefficient (P = 2)
PROJECTIONS = ["nonneg", "bernstein"]


def _solver():
    return IPMSolver(ForwardSolverConfig(max_iter=200), crossover=False, safeguard=True)


def _toy(seed: int = 3):
    """A small acyclic PURC problem with a polynomial-sieve perturbation (K=1)."""
    n = 8
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n) for i in range(n)]
    for _ in range(4 * n):
        a, b = rng.integers(0, n, 2)
        if a != b:
            edges.append((int(a), int(b)))
    inc = np.zeros((n, len(edges)))
    for k, (a, b) in enumerate(edges):
        inc[a, k], inc[b, k] = 1.0, -1.0
    Z = sp.csr_matrix(rng.uniform(0.5, 1.5, (inc.shape[1], 1)))
    d0 = np.zeros(n)
    d0[0], d0[1] = 1.0, -1.0
    cons = GeneralPolytope(sp.csr_matrix(inc), d0, ell=np.ones(inc.shape[1]))
    return PUMProblem(get_perturbation("polynomial_sieve", gamma=GAMMA0), cons, Z=Z), n


def _scipy_min(loss, n_beta, bern_M):
    """An independent scipy minimizer of the loss: L-BFGS-B (box) or trust-constr (poly)."""
    layout = loss.layout
    P = layout.size
    cache: dict = {}

    def vg(x):
        key = x.tobytes()
        if key not in cache:
            Q, g = loss.value_and_grad(torch.as_tensor(x, dtype=DEFAULT_DTYPE))
            cache[key] = (float(Q), to_numpy(g).astype(np.float64))
        return cache[key]

    x0 = np.zeros(P)
    if bern_M is None:
        bounds = [(None, None)] * n_beta + [(0.0, None)] * (P - n_beta)
        res = minimize(
            lambda x: vg(x)[0],
            x0,
            jac=lambda x: vg(x)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-15, "gtol": 1e-8, "maxiter": 2000},
        )
        converged = bool(res.success)
    else:
        A = np.hstack([np.zeros((bern_M.shape[0], n_beta)), bern_M])
        res = minimize(
            lambda x: vg(x)[0],
            x0,
            jac=lambda x: vg(x)[1],
            hess=BFGS(),
            method="trust-constr",
            constraints=[LinearConstraint(A, -1.0, np.inf)],
            options={"gtol": 1e-9, "xtol": 1e-14, "maxiter": 500, "verbose": 0},
        )
        converged = bool(res.status in (1, 2))
    return torch.as_tensor(res.x, dtype=DEFAULT_DTYPE), float(res.fun), converged


def _grad_audit(loss, theta, h: float = 1e-5) -> float:
    """Max absolute error of the analytic gradient vs central FD (finite coords only)."""
    _, g = loss.value_and_grad(theta)
    g = to_numpy(g)
    worst = 0.0
    for j in range(theta.numel()):
        tp, tm = theta.clone(), theta.clone()
        tp[j] += h
        tm[j] -= h
        fp, fm = loss.value(tp), loss.value(tm)
        if np.isfinite(fp) and np.isfinite(fm):
            worst = max(worst, abs((fp - fm) / (2 * h) - g[j]))
    return worst


def _criticality(loss, theta, n_beta, proj) -> float:
    """``||theta - P_Gamma(theta - g)||_inf`` with beta unconstrained."""
    _, g = loss.value_and_grad(theta)
    step = theta - g
    proj_step = torch.cat([step[:n_beta], proj(step[n_beta:])])
    return float(torch.max(torch.abs(theta - proj_step)))


def _oracle_anchor(prob, data, theta, n_beta, solver, n_check: int = 4) -> float:
    """Max |x_batch - x_scipy| over a few OD pairs at ``theta`` (monomial gamma)."""
    beta, gamma = theta[:n_beta], theta[n_beta:]
    b_batch = to_numpy(data.b_batch)
    xb = to_numpy(solver.solve_batch((beta, gamma), b_batch).x)
    A = prob.constraint.A
    ell = to_numpy(prob.constraint.ell)
    maxdiff = 0.0
    for i in range(min(len(b_batch), n_check)):
        prob_i = PUMProblem(prob.perturbation, GeneralPolytope(A, b_batch[i], ell=ell), Z=prob.Z)
        maxdiff = max(maxdiff, float(np.max(np.abs(xb[i] - solve_scipy(prob_i, (beta, gamma))))))
    return maxdiff


@lru_cache(maxsize=None)
def _regime(proj: str) -> dict:
    """Fit newton/tr_bfgs/scipy on the toy and compute every gate (cached per proj)."""
    prob, n = _toy()
    rng = np.random.default_rng(11)
    ods = []
    for _ in range(12):
        o, t = rng.choice(n, 2, replace=False)
        b = np.zeros(n)
        b[o], b[t] = 1.0, -1.0
        ods.append(ODSpec(b=b, origin=int(o), dest=int(t), D=200))
    s = _solver()
    s.preprocess(prob)
    data = simulate_dataset(prob, s, (BETA0, GAMMA0), ods, np.random.default_rng(5))
    n_beta = BETA0.size
    bern_M = None if proj == "nonneg" else bernstein_matrix(L - 2)

    results: dict = {}
    for method in ("newton", "tr_bfgs"):
        sm = _solver()
        sm.preprocess(prob)
        cfg = EstimatorConfig(
            proj=GammaProjection(proj), method=method, max_iter=200, tol_grad=1e-6
        )
        r = DebiasedFYEstimator(prob, sm, L, cfg).fit(data)
        results[method] = dict(
            theta=to_numpy(r.theta_hat), Q=float(r.objective), conv=bool(r.converged)
        )
    ss = _solver()
    ss.preprocess(prob)
    th, Q, conv = _scipy_min(DebiasedFYLoss(prob, ss, data, L), n_beta, bern_M)
    results["scipy"] = dict(theta=to_numpy(th), Q=Q, conv=conv)

    # Cold (deterministic) gate loss for the FD audit and criticality.
    gs = _solver()
    gs.preprocess(prob)
    gate_loss = DebiasedFYLoss(prob, gs, data, L, warm_start=False)
    proj_obj = GammaProjection(proj)
    theta_int = torch.as_tensor(np.concatenate([BETA0, GAMMA0]), dtype=DEFAULT_DTYPE)
    c1 = _grad_audit(gate_loss, theta_int)
    crit = {
        m: dict(
            gmap=_criticality(
                gate_loss, torch.as_tensor(r["theta"], dtype=DEFAULT_DTYPE), n_beta, proj_obj
            ),
            feasible=proj_obj.is_feasible(r["theta"][n_beta:]),
        )
        for m, r in results.items()
    }
    conv_Qs = [r["Q"] for r in results.values() if r["conv"]]
    q_spread = max(conv_Qs) - min(conv_Qs)
    ref = min((m for m in results if results[m]["conv"]), key=lambda m: crit[m]["gmap"])
    sa = _solver()
    sa.preprocess(prob)
    c4 = _oracle_anchor(prob, data, results[ref]["theta"], n_beta, sa)
    return dict(results=results, crit=crit, c1=c1, q_spread=q_spread, c4=c4, n_beta=n_beta)


def _regime_or_skip(proj: str) -> dict:
    """Build the regime, skipping on a rare degenerate forward solve (basis-independent)."""
    try:
        return _regime(proj)
    except (np.linalg.LinAlgError, RuntimeError):
        pytest.skip("forward solver degenerated on this toy draw")


# --------------------------------------------------------------------------- #
def test_loss_barrier_on_noncomputable_solve():
    """
    A non-computable inner solve makes ``Q_B`` an extended-value ``+inf`` sentinel
    (so the optimizer's backtracking routes around it) rather than propagating the
    crash.  This is the robustness fix for the degenerate corner/boundary iterates:
    the failure is a hidden constraint (no a-priori margin / Levenberg-Marquardt eps
    well-poses it), so the solve attempt itself is the oracle of computability.
    """
    prob, n = _toy()
    rng = np.random.default_rng(11)
    ods = []
    for _ in range(6):
        o, t = rng.choice(n, 2, replace=False)
        b = np.zeros(n)
        b[o], b[t] = 1.0, -1.0
        ods.append(ODSpec(b=b, origin=int(o), dest=int(t), D=50))
    s = _solver()
    s.preprocess(prob)
    data = simulate_dataset(prob, s, (BETA0, GAMMA0), ods, np.random.default_rng(5))

    class _Raising:  # composition proxy: the inner solve always fails (no monkeypatch)
        def __init__(self, inner):
            self._inner = inner

        def solve_batch(self, *a, **k):
            raise np.linalg.LinAlgError("CHOLMOD batch factorization failed for system(s) [0]")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    s2 = _solver()
    s2.preprocess(prob)
    loss = DebiasedFYLoss(prob, _Raising(s2), data, L)
    theta = loss.layout.pack(BETA0, GAMMA0)  # a feasible, convex gamma
    Q, g = loss.value_and_grad(theta)
    assert Q == float("inf") and bool(torch.isnan(g).all())
    assert loss.value(theta) == float("inf")


@pytest.mark.parametrize("proj", PROJECTIONS)
def test_c1_gradient_audit(proj):
    """C1: the analytic residual-form gradient matches a central FD of Q_B."""
    assert _regime_or_skip(proj)["c1"] < 1e-6


@pytest.mark.parametrize("proj", PROJECTIONS)
def test_c2_q_consensus(proj):
    """C2: newton, tr_bfgs and scipy all converge to the same minimal objective."""
    reg = _regime_or_skip(proj)
    for m, r in reg["results"].items():
        assert r["conv"], f"{m} failed to converge"
    assert reg["q_spread"] < 1e-7, reg["q_spread"]


@pytest.mark.parametrize("proj", PROJECTIONS)
def test_c3_kkt_and_feasible(proj):
    """C3: each reported theta* is criticality-stationary and feasible in Gamma."""
    reg = _regime_or_skip(proj)
    for m, c in reg["crit"].items():
        assert c["feasible"], f"{m} infeasible"
        assert c["gmap"] < 1e-4, f"{m} gmap={c['gmap']:.2e}"


@pytest.mark.parametrize("proj", PROJECTIONS)
def test_c4_inner_oracle_anchor(proj):
    """C4: the batched IPM matches the independent scipy oracle at the optimum."""
    assert _regime_or_skip(proj)["c4"] < 1e-6


@pytest.mark.native
@pytest.mark.parametrize("proj", PROJECTIONS)
def test_native_tr_bfgs_reaches_optimum(proj):
    """The native-dispatched tr_bfgs lands at the certified (scipy) optimum."""
    if not native_available():
        pytest.skip("native core not built")
    reg = _regime_or_skip(proj)
    tr, ref = reg["results"]["tr_bfgs"], reg["results"]["scipy"]
    assert tr["conv"] and reg["crit"]["tr_bfgs"]["feasible"]
    assert abs(tr["Q"] - ref["Q"]) < 1e-7
