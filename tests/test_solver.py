"""
End-to-end tests for the RegularizedSSNSolver (torch-native).

Parametrized over closed-form perturbations and constraint geometries:
  * full-rank equality (a simplex-style sum constraint),
  * rank-deficient node-arc incidence (a network flow polytope).

Checks, for each combination: convergence + primal feasibility, agreement with
the CVXPY oracle, gauge-invariance of ``x*`` under shifted warm-start multipliers,
KKT stationarity, warm-start consistency, and the conjugate value.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from purcsolver import PUMProblem, SSNConfig
from purcsolver.constraints import GeneralPolytope
from purcsolver.perturbations import get_perturbation
from purcsolver.solvers import RegularizedSSNSolver
from purcsolver.utils.torch_compat import to_numpy

cvxpy = pytest.importorskip("cvxpy")
from purcsolver.oracle import solve_cvxpy  # noqa: E402

KERNELS = ["quadratic", "entropy", "logit_entropy", "modified_entropy"]
NO_GAMMA = torch.zeros(0, dtype=torch.float64)


def _simplex_problem(n=6, seed=0):
    """Full-rank single equality sum(x)=1 on the unit box."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    A = sp.csr_matrix(np.ones((1, n)))
    return GeneralPolytope(A, np.array([1.0]), lo=0.0, hi=1.0), v


def _network_problem(seed=1):
    """Rank-deficient node-arc incidence; one unit of flow node 0 -> node 3."""
    inc = np.array(
        [[1, 0, 0, 1, 0], [-1, 1, 0, 0, 1], [0, -1, 1, -1, 0], [0, 0, -1, 0, -1]],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    ell = rng.uniform(0.5, 2.0, inc.shape[1])
    v = -rng.uniform(0.2, 1.5, inc.shape[1])  # negative utilities
    poly = GeneralPolytope(sp.csr_matrix(inc), np.array([1.0, 0.0, 0.0, -1.0]), ell=ell)
    return poly, v


CONSTRAINTS = {"simplex": _simplex_problem, "network": _network_problem}


@pytest.fixture(params=[(k, c) for k in KERNELS for c in CONSTRAINTS])
def case(request):
    kname, cname = request.param
    poly, v = CONSTRAINTS[cname]()
    prob = PUMProblem(get_perturbation(kname), poly)
    return kname, cname, prob, v


def _solve(prob, v, **cfg):
    solver = RegularizedSSNSolver(SSNConfig(tol=1e-10, **cfg))
    solver.preprocess(prob)
    return solver, solver.solve((v, NO_GAMMA))


def test_converges_and_feasible(case):
    _, _, prob, v = case
    _, res = _solve(prob, v)
    assert res.success
    feas = float((prob.constraint.matvec(res.x) - prob.constraint.b).abs().max())
    assert feas < 1e-8


def test_matches_cvxpy_oracle(case):
    _, _, prob, v = case
    _, res = _solve(prob, v)
    x_oracle = solve_cvxpy(prob, (v, NO_GAMMA))
    np.testing.assert_allclose(to_numpy(res.x), x_oracle, atol=1e-6)


def test_gauge_invariance_of_primal(case):
    """Shifting the warm-start multipliers must not change x* (only lambda)."""
    _, _, prob, v = case
    solver, res0 = _solve(prob, v)
    k = prob.constraint.num_constraints
    res_shift = solver.solve((v, NO_GAMMA), lam0=np.full(k, 7.0))
    np.testing.assert_allclose(to_numpy(res0.x), to_numpy(res_shift.x), atol=1e-7)


def test_kkt_stationarity_on_active_set(case):
    """On interior coords, ell_i h'(x_i) = v_i + (A^T lambda)_i."""
    _, _, prob, v = case
    _, res = _solve(prob, v)
    c = prob.constraint
    pert = prob.perturbation
    eta = (prob.utility(v) + c.rmatvec(res.lam)) / c.ell
    x_hat, interior = pert.primal_recovery(eta, c.lo, c.hi, NO_GAMMA)
    np.testing.assert_allclose(to_numpy(x_hat), to_numpy(res.x), atol=1e-9)
    resid = c.ell * pert.hprime(res.x, NO_GAMMA) - (prob.utility(v) + c.rmatvec(res.lam))
    assert float(resid[interior].abs().max()) < 1e-6


def test_warm_start_consistency(case):
    """Cold and warm starts reach the same primal solution."""
    _, _, prob, v = case
    _, res_cold = _solve(prob, v)
    solver_warm = RegularizedSSNSolver(SSNConfig(tol=1e-10, warm_start=True))
    solver_warm.preprocess(prob)
    solver_warm.solve((v, NO_GAMMA))  # populate warm start
    res_warm = solver_warm.solve((v, NO_GAMMA))
    np.testing.assert_allclose(to_numpy(res_cold.x), to_numpy(res_warm.x), atol=1e-7)
    assert res_warm.nit <= res_cold.nit  # warm restart should not be slower


def test_conjugate_value(case):
    _, _, prob, v = case
    _, res = _solve(prob, v)
    c = prob.constraint
    expected = float(prob.utility(v) @ res.x - c.ell @ prob.perturbation.h(res.x, NO_GAMMA))
    assert res.conjugate == pytest.approx(expected, abs=1e-9)
