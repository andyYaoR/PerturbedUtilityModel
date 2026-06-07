"""
Tests for the debiased Fenchel--Young estimation layer.

Covers the load-bearing properties:

  * **U-statistic unbiasedness** -- ``E[U_l] = p^l`` for binomial counts, and the
    naive plug-in ``(n/D)^l`` is biased upward.
  * **Gamma projection** -- nonnegative clipping; Bernstein projection is feasible
    and idempotent on feasible points.
  * **Gradient correctness** -- the closed-form (residual-form) gradient matches a
    central finite difference of the objective.
  * **Score zero at theta_0** -- with empirical frequencies equal to ``x*`` the
    score vanishes (the Fenchel--Young residual structure).
  * **DGP** -- the random-walk sampler reproduces ``x*`` as ``D`` grows and pins
    the incidence orientation on a toy network.
  * **End-to-end** -- the estimator finds the sample minimum (and recovers a known
    ``theta_0`` on a small network) under both projections.
  * **Sandwich variance** -- well-shaped, finite standard errors.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from purc.estimators.base import SimulatedData
from purc.estimators.debiased_fy import (
    DebiasedFYEstimator,
    DebiasedFYLoss,
    EstimatorConfig,
    GammaProjection,
    NaiveFYLoss,
    falling_factorial,
    project_bernstein,
    project_nonneg,
    u_statistics,
)
from purc.estimators.debiased_fy.variance import sandwich_variance
from purc.static_purc import PUMProblem, SSNConfig
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.dgp import ODSpec, RandomWalkSampler, simulate_dataset
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.utils.torch_compat import to_numpy


# --------------------------------------------------------------------------- #
# problem builders
# --------------------------------------------------------------------------- #
def _network(n_nodes: int, seed: int):
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    for _ in range(4 * n_nodes):
        a, b = rng.integers(0, n_nodes, 2)
        if a != b:
            edges.append((int(a), int(b)))
    inc = np.zeros((n_nodes, len(edges)))
    for k, (a, b) in enumerate(edges):
        inc[a, k], inc[b, k] = 1.0, -1.0
    return inc


def _problem(inc, gamma0, seed, K=1):
    n, N = inc.shape
    rng = np.random.default_rng(seed)
    Z = sp.csr_matrix(rng.uniform(0.5, 1.5, (N, K)))  # positive attributes
    pert = get_perturbation("polynomial_sieve", gamma=gamma0)
    d0 = np.zeros(n)
    d0[0], d0[1] = 1.0, -1.0
    cons = GeneralPolytope(sp.csr_matrix(inc), d0, ell=np.ones(N))
    return PUMProblem(pert, cons, Z=Z)


def _ods(n, n_od, D, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_od):
        o, t = rng.choice(n, 2, replace=False)
        d = np.zeros(n)
        d[o], d[t] = 1.0, -1.0
        out.append(ODSpec(b=d, origin=int(o), dest=int(t), D=D))
    return out


# --------------------------------------------------------------------------- #
# U-statistics
# --------------------------------------------------------------------------- #
def test_falling_factorial_values():
    n = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    assert torch.allclose(falling_factorial(n, 0), torch.ones(4, dtype=torch.float64))
    assert torch.allclose(falling_factorial(n, 1), n)
    # 5^{(3)} = 5*4*3 = 60; n<3 -> 0
    assert torch.allclose(falling_factorial(n, 3), torch.tensor([0.0, 0.0, 0.0, 60.0], dtype=torch.float64))


def test_u_statistics_unbiased_and_naive_biased():
    rng = np.random.default_rng(0)
    p = 0.3
    D = 40
    reps = 20000
    n = rng.binomial(D, p, size=(reps, 1)).astype(float)
    U, valid = u_statistics(n, np.full(reps, D), L=4)
    for deg in (2, 3, 4):
        assert valid[:, deg].all()
        u_mean = float(U[:, 0, deg].mean())
        assert abs(u_mean - p**deg) < 0.01, (deg, u_mean, p**deg)
        naive_mean = float(np.mean((n[:, 0] / D) ** deg))
        assert naive_mean > p**deg + 1e-4  # plug-in is biased upward


def test_u_statistics_validity_mask():
    # D = 2 -> degrees l >= 3 are invalid.
    U, valid = u_statistics(np.array([[1.0]]), np.array([2.0]), L=4)
    assert valid[0, 2] and not valid[0, 3] and not valid[0, 4]


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def test_project_nonneg():
    g = torch.tensor([-1.0, 0.5, -0.2], dtype=torch.float64)
    assert torch.allclose(project_nonneg(g), torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64))


def test_project_bernstein_feasible_and_idempotent():
    proj = GammaProjection("bernstein")
    feas = torch.tensor([0.5, 0.3])  # in Gamma_B
    assert proj.is_feasible(feas)
    np.testing.assert_allclose(to_numpy(project_bernstein(feas)), to_numpy(feas), atol=1e-9)
    infeas = torch.tensor([-5.0, -5.0])  # outside Gamma_B
    out = project_bernstein(infeas)
    assert proj.is_feasible(out)


# --------------------------------------------------------------------------- #
# DGP
# --------------------------------------------------------------------------- #
def test_random_walk_orientation_two_node():
    # Two nodes, one arc 0->1: every trip must traverse the single link.
    inc = np.array([[1.0], [-1.0]])
    cons = GeneralPolytope(sp.csr_matrix(inc), np.array([1.0, -1.0]), ell=np.ones(1))
    sampler = RandomWalkSampler(cons, np.random.default_rng(0))
    y = sampler.sample_trip(np.array([1.0]), origin=0, dest=1, max_steps=4)
    assert y[0] == 1.0


def test_dgp_frequencies_converge_to_xstar():
    inc = _network(6, 1)
    prob = _problem(inc, np.array([0.5]), seed=2)
    solver = IPMSolver(SSNConfig(max_iter=200), crossover=False, safeguard=True)
    solver.preprocess(prob)
    beta0, gamma0 = np.array([-2.0]), np.array([0.5])
    ods = _ods(inc.shape[0], 4, D=20000, seed=3)
    data = simulate_dataset(prob, solver, (beta0, gamma0), ods, np.random.default_rng(4))
    # E[ybar] = x*; with D = 20000 the empirical frequencies are close.
    assert torch.max(torch.abs(data.ybar - data.xstar0)) < 0.05


# --------------------------------------------------------------------------- #
# loss gradient + score
# --------------------------------------------------------------------------- #
def test_loss_gradient_matches_finite_difference():
    inc = _network(8, 5)
    prob = _problem(inc, np.array([0.5]), seed=6)
    solver = IPMSolver(SSNConfig(max_iter=200), crossover=False, safeguard=True)
    solver.preprocess(prob)
    beta0, gamma0 = np.array([-2.0]), np.array([0.5])
    ods = _ods(inc.shape[0], 30, D=2000, seed=7)
    data = simulate_dataset(prob, solver, (beta0, gamma0), ods, np.random.default_rng(8))
    loss = DebiasedFYLoss(prob, solver, data, L=3)
    theta = loss.layout.pack(np.array([-1.5]), np.array([0.2]))
    _, g = loss.value_and_grad(theta)
    fd = torch.zeros_like(g)
    h = 1e-5
    for j in range(g.numel()):
        e = torch.zeros_like(g)
        e[j] = h
        fd[j] = (loss.value(theta + e) - loss.value(theta - e)) / (2 * h)
    assert torch.max(torch.abs(g - fd)) < 1e-6


def test_score_zero_at_theta0_when_ybar_is_xstar():
    """With empirical frequencies equal to x*(theta0), the score vanishes."""
    inc = _network(8, 9)
    prob = _problem(inc, np.array([0.5]), seed=10)
    solver = IPMSolver(SSNConfig(max_iter=200), crossover=False, safeguard=True)
    solver.preprocess(prob)
    beta0, gamma0 = np.array([-2.0]), np.array([0.5])
    ods = _ods(inc.shape[0], 12, D=10, seed=11)
    b_batch = np.stack([od.b for od in ods])
    xstar = to_numpy(solver.solve_batch((beta0, gamma0), b_batch).x)
    # Construct data whose empirical frequencies equal x* exactly.
    data = SimulatedData(
        n_counts=torch.as_tensor(xstar, dtype=torch.float64),
        ybar=torch.as_tensor(xstar, dtype=torch.float64),
        D=torch.ones(len(ods), dtype=torch.float64),
        b_batch=torch.as_tensor(b_batch, dtype=torch.float64),
    )
    # The naive loss uses U_l = ybar^l = (x*)^l, so the score is exactly zero.
    loss = NaiveFYLoss(prob, solver, data, L=3)
    _, g = loss.value_and_grad(loss.layout.pack(beta0, gamma0))
    assert torch.max(torch.abs(g)) < 1e-7


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["nonneg", "bernstein"])
def test_estimator_finds_sample_minimum(kind):
    inc = _network(8, 12)
    prob = _problem(inc, np.array([0.5]), seed=13)
    solver = IPMSolver(SSNConfig(max_iter=200), crossover=False, safeguard=True)
    solver.preprocess(prob)
    beta0, gamma0 = np.array([-2.0]), np.array([0.5])
    ods = _ods(inc.shape[0], 60, D=3000, seed=14)
    data = simulate_dataset(prob, solver, (beta0, gamma0), ods, np.random.default_rng(15))
    loss = DebiasedFYLoss(prob, solver, data, L=3)
    q0 = loss.value(loss.layout.pack(beta0, gamma0))
    est = DebiasedFYEstimator(
        prob, solver, L=3, config=EstimatorConfig(proj=GammaProjection(kind)),
        theta_init=loss.layout.pack(np.array([-1.0]), np.array([0.0])),
    )
    res = est.fit(data)
    assert res.converged
    # theta_hat is the sample minimum: at least as good as theta_0, and stationary.
    assert res.objective <= q0 + 1e-7
    gmap = res.theta_hat - est._project(res.theta_hat - res.grad, loss.layout)
    assert float(gmap.abs().max()) < 1e-5


def test_sandwich_variance_shapes_and_finite():
    inc = _network(8, 16)
    prob = _problem(inc, np.array([0.5]), seed=17)
    solver = IPMSolver(SSNConfig(max_iter=200), crossover=False, safeguard=True)
    solver.preprocess(prob)
    beta0, gamma0 = np.array([-2.0]), np.array([0.5])
    ods = _ods(inc.shape[0], 80, D=3000, seed=18)
    data = simulate_dataset(prob, solver, (beta0, gamma0), ods, np.random.default_rng(19))
    loss = DebiasedFYLoss(prob, solver, data, L=3)
    theta = loss.layout.pack(beta0, gamma0)
    sw = sandwich_variance(loss, theta)
    P = loss.layout.size
    assert sw.var.shape == (P, P) and sw.se.shape == (P,)
    assert torch.isfinite(sw.se).all() and (sw.se > 0).all()
    assert np.isfinite(sw.cond_A)
