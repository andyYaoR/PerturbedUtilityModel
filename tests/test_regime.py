"""
Tests for the provable solver-regime rule and the unified :class:`AutoSolver`.

The regime is selected by a *provable* property of the (kernel, box) pair --
:meth:`SeparablePerturbation.admits_primal_interior` -- not by catching a runtime
failure.  These tests pin:

  * the classification itself (where ``h'`` diverges relative to the box),
  * the IPM's precondition assertion on Legendre-type kernels,
  * that :class:`AutoSolver` routes by the rule, and
  * that the unified solver (and the barrier-continuation backstop) converge to
    the CVXPY oracle on every kernel, including the Legendre ones.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from purcsolver import PUMProblem, SSNConfig
from purcsolver.constraints import GeneralPolytope
from purcsolver.perturbations import get_perturbation
from purcsolver.solvers import AutoSolver, BarrierContinuationSolver, IPMSolver

cvxpy = pytest.importorskip("cvxpy")
from purcsolver.oracle import solve_cvxpy  # noqa: E402
from purcsolver.utils.torch_compat import to_numpy  # noqa: E402

NO_GAMMA = torch.zeros(0, dtype=torch.float64)

# kernel -> expected regime under the provable rule on the standard box [0, 1]
SMOOTH_ON_BOX = ["quadratic", "modified_entropy", "polynomial_sieve"]
LEGENDRE = ["entropy", "logit_entropy"]


def _simplex_problem(kernel, n=6, seed=0):
    """Full-rank single equality sum(x)=1 on the unit box."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    A = sp.csr_matrix(np.ones((1, n)))
    poly = GeneralPolytope(A, np.array([1.0]), lo=0.0, hi=1.0)
    return PUMProblem(get_perturbation(kernel), poly), v


def _network_problem(kernel, seed=1):
    """Rank-deficient node-arc incidence; one unit of flow node 0 -> node 3."""
    inc = np.array(
        [[1, 0, 0, 1, 0], [-1, 1, 0, 0, 1], [0, -1, 1, -1, 0], [0, 0, -1, 0, -1]],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    ell = rng.uniform(0.5, 2.0, inc.shape[1])
    v = -rng.uniform(0.2, 1.5, inc.shape[1])
    poly = GeneralPolytope(sp.csr_matrix(inc), np.array([1.0, 0.0, 0.0, -1.0]), ell=ell)
    return PUMProblem(get_perturbation(kernel), poly), v


CONSTRAINTS = {"simplex": _simplex_problem, "network": _network_problem}


# --------------------------------------------------------------------------- #
# The provable classification itself.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kernel", SMOOTH_ON_BOX)
def test_smooth_on_box_admits_primal_interior(kernel):
    """Quadratic / modified-entropy / sieve have finite h' on [0, 1]."""
    pert = (
        get_perturbation("polynomial_sieve", gamma=np.array([0.3, 0.1]))
        if kernel == "polynomial_sieve"
        else get_perturbation(kernel)
    )
    assert pert.admits_primal_interior(np.zeros(4), np.ones(4)) is True


@pytest.mark.parametrize("kernel", LEGENDRE)
def test_legendre_kernels_excluded_on_unit_box(kernel):
    """Shannon / logit entropy have a divergent h' at a face of [0, 1]."""
    assert get_perturbation(kernel).admits_primal_interior(np.zeros(4), np.ones(4)) is False


def test_rule_depends_on_box_not_kernel_name():
    """
    The rule is structural: it tracks where h' diverges relative to the box.

    Modified entropy (h' singular at -1) is primal-interior on [0, 1] but not on a
    box whose lower face touches -1; Shannon entropy (h' singular at 0) becomes
    primal-interior once the box is pulled strictly inside (0, 1).
    """
    me = get_perturbation("modified_entropy")
    assert me.admits_primal_interior(np.zeros(3), np.ones(3)) is True
    assert me.admits_primal_interior(np.full(3, -1.0), np.ones(3)) is False
    assert me.admits_primal_interior(np.full(3, -0.5), np.ones(3)) is True

    shannon = get_perturbation("entropy")
    assert shannon.admits_primal_interior(np.zeros(3), np.ones(3)) is False
    assert shannon.admits_primal_interior(np.full(3, 0.01), np.full(3, 0.99)) is True


# --------------------------------------------------------------------------- #
# Methods assert their precondition instead of catching failures.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kernel", LEGENDRE)
def test_ipm_rejects_legendre_kernel(kernel):
    """The primal IPM raises a clear precondition error (no NaN) on Legendre kernels."""
    prob, _ = _network_problem(kernel)
    solver = IPMSolver(SSNConfig(tol=1e-9))
    with pytest.raises(ValueError, match="essentially smooth"):
        solver.preprocess(prob)


@pytest.mark.parametrize("kernel", SMOOTH_ON_BOX)
def test_ipm_accepts_smooth_on_box_kernel(kernel):
    """The IPM preprocesses fine for kernels finite on the closed box."""
    prob, _ = (
        (_network_problem(kernel))
        if kernel != "polynomial_sieve"
        else (_network_sieve())
    )
    IPMSolver(SSNConfig(tol=1e-9)).preprocess(prob)  # must not raise


def _network_sieve(seed=1):
    inc = np.array(
        [[1, 0, 0, 1, 0], [-1, 1, 0, 0, 1], [0, -1, 1, -1, 0], [0, 0, -1, 0, -1]],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    ell = rng.uniform(0.5, 2.0, inc.shape[1])
    v = -rng.uniform(0.2, 1.5, inc.shape[1])
    poly = GeneralPolytope(sp.csr_matrix(inc), np.array([1.0, 0.0, 0.0, -1.0]), ell=ell)
    return PUMProblem(get_perturbation("polynomial_sieve", gamma=np.array([0.3, 0.1])), poly), v


# --------------------------------------------------------------------------- #
# AutoSolver routes by the rule and converges everywhere.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kernel", SMOOTH_ON_BOX + LEGENDRE)
def test_auto_routes_by_rule(kernel):
    """AutoSolver picks the IPM regime for smooth-on-box kernels, SSN for Legendre."""
    if kernel == "polynomial_sieve":
        prob, _ = _network_sieve()
    else:
        prob, _ = _network_problem(kernel)
    solver = AutoSolver(SSNConfig(tol=1e-9))
    solver.preprocess(prob)
    assert solver.regime == ("ipm" if kernel in SMOOTH_ON_BOX else "ssn")


@pytest.mark.parametrize("cname", list(CONSTRAINTS))
@pytest.mark.parametrize("kernel", ["quadratic", "entropy", "logit_entropy", "modified_entropy"])
def test_auto_matches_cvxpy_oracle(kernel, cname):
    """The unified solver converges and matches the CVXPY oracle on every kernel."""
    prob, v = CONSTRAINTS[cname](kernel)
    solver = AutoSolver(SSNConfig(tol=1e-10, max_iter=200))
    solver.preprocess(prob)
    res = solver.solve((v, NO_GAMMA))
    assert res.success
    feas = float((prob.constraint.matvec(res.x) - prob.constraint.b).abs().max())
    assert feas < 1e-7
    x_oracle = solve_cvxpy(prob, (v, NO_GAMMA))
    assert np.allclose(to_numpy(res.x).ravel(), x_oracle, atol=1e-6)


@pytest.mark.parametrize("cname", list(CONSTRAINTS))
@pytest.mark.parametrize("kernel", ["quadratic", "entropy", "logit_entropy", "modified_entropy"])
def test_barrier_backstop_matches_oracle(kernel, cname):
    """The barrier-continuation backstop is robust on every kernel (incl. Legendre)."""
    prob, v = CONSTRAINTS[cname](kernel)
    solver = BarrierContinuationSolver(SSNConfig(tol=1e-10, max_iter=300))
    solver.preprocess(prob)
    res = solver.solve((v, NO_GAMMA))
    assert res.success
    x_oracle = solve_cvxpy(prob, (v, NO_GAMMA))
    assert np.allclose(to_numpy(res.x).ravel(), x_oracle, atol=1e-6)
