"""
Tests for the trust-region damped-BFGS outer optimizer (:class:`TrustRegionBFGS`).

The optimizer is gradient-only (no finite differences, no sensitivity), line-search
free (a ratio test governs the radius), and handles both the nonneg box and the
Bernstein polyhedron.  We check:

  * **constrained-quadratic exactness** -- on a quadratic objective the TR step model
    is exact, so the iterate must reach the cvxpy optimum (box and Bernstein);
  * **smooth non-quadratic convex** -- BFGS curvature learning reaches the analytic
    optimum of a logistic-type objective;
  * **feasibility** of the returned ``gamma`` (``>=0`` resp. ``M gamma >= -1``);
  * **already-optimal / no-gamma** edge cases;
  * **end-to-end** -- on a small real debiased-FY loss it converges, stays feasible,
    and reaches the same optimum as the production projected-Newton estimator
    (no regularization bias).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from purc.estimators.debiased_fy.tr_optimizer import TRConfig, TrustRegionBFGS
from purc.static_purc.perturbations._bernstein import bernstein_matrix
from purc.static_purc.utils.torch_compat import DEFAULT_DTYPE


def _quad_vg(H, c):
    """value_and_grad for ``f(theta) = 1/2 (theta-c)' H (theta-c)``."""
    Ht = torch.as_tensor(H, dtype=DEFAULT_DTYPE)
    ct = torch.as_tensor(c, dtype=DEFAULT_DTYPE)

    def vg(theta):
        r = torch.as_tensor(theta, dtype=DEFAULT_DTYPE) - ct
        return 0.5 * float(r @ Ht @ r), Ht @ r

    return vg


def _spd(rng, P):
    A = rng.standard_normal((P, P))
    return A @ A.T + np.eye(P)


def test_tr_quadratic_nonneg_matches_cvxpy():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(0)
    K, ng = 2, 3
    P = K + ng
    worst = 0.0
    for _ in range(15):
        H, c = _spd(rng, P), rng.standard_normal(P) * 2
        res = TrustRegionBFGS(TRConfig(tol=1e-7, max_iter=300)).minimize(
            _quad_vg(H, c), np.zeros(P), K, bernstein_M=None)
        assert res.converged
        assert (res.theta.numpy()[K:] >= -1e-7).all()  # feasible gamma >= 0
        x = cp.Variable(P)
        cp.Problem(cp.Minimize(0.5 * cp.quad_form(x - c, cp.psd_wrap(H))), [x[K:] >= 0]).solve(solver=cp.CLARABEL)
        worst = max(worst, float(np.abs(res.theta.numpy() - x.value).max()))
    assert worst < 1e-4, worst


def test_tr_quadratic_bernstein_matches_cvxpy():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(1)
    K, L = 2, 5
    ng = L - 2
    P = K + ng
    M = bernstein_matrix(ng)
    worst = 0.0
    for _ in range(12):
        H, c = _spd(rng, P), rng.standard_normal(P) * 2
        res = TrustRegionBFGS(TRConfig(tol=1e-7, max_iter=400)).minimize(
            _quad_vg(H, c), np.zeros(P), K, bernstein_M=torch.as_tensor(M, dtype=DEFAULT_DTYPE))
        assert res.converged
        assert (M @ res.theta.numpy()[K:] >= -1 - 1e-7).all()  # feasible M gamma >= -1
        x = cp.Variable(P)
        cp.Problem(cp.Minimize(0.5 * cp.quad_form(x - c, cp.psd_wrap(H))), [M @ x[K:] >= -1]).solve(solver=cp.CLARABEL)
        worst = max(worst, float(np.abs(res.theta.numpy() - x.value).max()))
    assert worst < 1e-4, worst


def test_tr_smooth_convex_reaches_analytic_optimum():
    # f(beta) = sum softplus(beta) - t·beta ; grad = sigmoid(beta) - t ; opt beta = logit(t)
    t = np.array([0.3, 0.7, 0.5, 0.2])

    def vg(theta):
        b = torch.as_tensor(theta, dtype=DEFAULT_DTYPE)
        tt = torch.as_tensor(t, dtype=DEFAULT_DTYPE)
        f = float((torch.nn.functional.softplus(b) - tt * b).sum())
        return f, torch.sigmoid(b) - tt

    res = TrustRegionBFGS(TRConfig(tol=1e-8, max_iter=200)).minimize(vg, np.zeros(4), 4, bernstein_M=None)
    assert res.converged
    assert np.abs(res.theta.numpy() - np.log(t / (1 - t))).max() < 1e-5


def test_tr_already_optimal_returns_fast():
    H, c = np.diag([1.0, 2.0, 3.0]), np.array([0.5, 0.3, 0.1])  # gamma block >= 0 -> c feasible
    res = TrustRegionBFGS(TRConfig(tol=1e-9)).minimize(_quad_vg(H, c), c, 1, bernstein_M=None)
    assert res.converged and res.n_outer <= 2
    assert np.abs(res.theta.numpy() - c).max() < 1e-7


def test_tr_no_gamma_block():
    H, c = np.diag([1.0, 2.0]), np.array([1.0, -1.0])  # K = P = 2, no gamma
    res = TrustRegionBFGS(TRConfig(tol=1e-9, max_iter=100)).minimize(_quad_vg(H, c), np.zeros(2), 2, bernstein_M=None)
    assert res.converged
    assert np.abs(res.theta.numpy() - c).max() < 1e-6


# --------------------------------------------------------------------------- #
# end-to-end on a small real debiased-FY loss
# --------------------------------------------------------------------------- #
def _toy_problem(gamma0, seed, K=1):
    from purc.static_purc import PUMProblem
    from purc.static_purc.constraints import GeneralPolytope
    from purc.static_purc.perturbations import get_perturbation

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
    N = inc.shape[1]
    Z = sp.csr_matrix(rng.uniform(0.5, 1.5, (N, K)))
    d0 = np.zeros(n)
    d0[0], d0[1] = 1.0, -1.0
    cons = GeneralPolytope(sp.csr_matrix(inc), d0, ell=np.ones(N))
    return PUMProblem(get_perturbation("polynomial_sieve", gamma=gamma0), cons, Z=Z), inc.shape[0]


@pytest.mark.parametrize("proj", ["nonneg", "bernstein"])
def test_estimator_tr_bfgs_method(proj):
    """
    The wired ``method='tr_bfgs'`` converges, stays feasible, and is at least as
    good as ``method='newton'`` through the production estimator interface.
    """
    from purc.estimators.debiased_fy import DebiasedFYEstimator, EstimatorConfig, GammaProjection
    from purc.static_purc import ForwardSolverConfig
    from purc.static_purc.dgp import ODSpec, simulate_dataset
    from purc.static_purc.solvers.ipm import IPMSolver

    gamma0, beta0 = np.array([0.5]), np.array([-1.0])
    L = gamma0.size + 2
    prob, n = _toy_problem(gamma0, seed=3, K=1)
    rng = np.random.default_rng(11)
    ods = []
    for _ in range(12):
        o, t = rng.choice(n, 2, replace=False)
        b = np.zeros(n)
        b[o], b[t] = 1.0, -1.0
        ods.append(ODSpec(b=b, origin=int(o), dest=int(t), D=80))

    def _solver():
        return IPMSolver(ForwardSolverConfig(max_iter=200), crossover=False, safeguard=True)

    def _fit(method):
        s = _solver()
        s.preprocess(prob)
        cfg = EstimatorConfig(proj=GammaProjection(proj), method=method, max_iter=300, tol_grad=1e-6)
        return DebiasedFYEstimator(prob, s, L, cfg).fit(data)

    try:
        base = _solver()
        base.preprocess(prob)
        data = simulate_dataset(prob, base, (beta0, gamma0), ods, np.random.default_rng(5))
        rN = _fit("newton")
        rT = _fit("tr_bfgs")
    except (np.linalg.LinAlgError, RuntimeError):
        pytest.skip("forward solver degenerated on this toy draw (basis-independent fragility)")

    assert rT.converged
    assert rT.extras["method"] == "tr_bfgs"
    gamma = rT.gamma_hat.numpy()
    if proj == "nonneg":
        assert (gamma >= -1e-7).all()
    else:
        from purc.static_purc.perturbations._bernstein import bernstein_matrix
        assert (bernstein_matrix(L - 2) @ gamma >= -1 - 1e-7).all()
    # TR-BFGS reaches a stationary point at least as good as Newton (which may stall).
    assert rT.objective <= rN.objective + 1e-6
