"""Tests for fixed-mu log-barrier PURC primitives."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from purc.static_purc import PUMProblem
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers.barrier import (
    BarrierRecoveryConfig,
    barrier_dual_objective,
    recover_barrier_primal,
)
from purc.static_purc.utils.torch_compat import as_tensor


def _problem(kernel: str) -> tuple[PUMProblem, torch.Tensor, torch.Tensor]:
    """Build a small finite-box problem for barrier primitive tests."""
    A = sp.csr_matrix(np.array([[1.0, -1.0, 0.0, 1.0], [0.0, 1.0, -1.0, 0.0]]))
    ell = np.array([0.7, 1.4, 0.9, 1.8])
    poly = GeneralPolytope(A, np.zeros(2), ell=ell, validate=False)
    if kernel == "sieve":
        gamma = torch.tensor([0.3, 0.1], dtype=torch.float64)
        pert = get_perturbation("polynomial_sieve", gamma=gamma)
    else:
        gamma = torch.zeros(0, dtype=torch.float64)
        pert = get_perturbation(kernel)
    return PUMProblem(pert, poly), torch.tensor([-4.0, 0.5, 1.0, -2.0]), gamma


@pytest.mark.parametrize("kernel", ["quadratic", "entropy", "sieve"])
def test_barrier_recovery_solves_scalar_kkt(kernel):
    """Recovered coordinates solve the fixed-mu scalar stationarity equation."""
    prob, v, gamma = _problem(kernel)
    lam = torch.tensor([0.7, -0.2], dtype=torch.float64)
    mu = 0.35
    x, weight = recover_barrier_primal(prob, v, lam, gamma, mu)
    c = prob.constraint
    y = v + c.rmatvec(lam)

    assert bool((x > c.lo).all() and (x < c.hi).all())
    assert bool(torch.isfinite(weight).all() and (weight > 0).all())

    residual = (
        c.ell * prob.perturbation.hprime(x, gamma)
        - y
        - mu / (x - c.lo)
        + mu / (c.hi - x)
    )
    assert float(residual.abs().max()) < 1e-8


def test_barrier_dual_gradient_matches_feasibility_residual():
    """Finite differences of the barrier dual equal ``(A x - b)^T d``."""
    prob, v, gamma = _problem("quadratic")
    lam = torch.tensor([0.4, -0.6], dtype=torch.float64)
    b = torch.tensor([0.2, -0.1], dtype=torch.float64)
    direction = torch.tensor([0.3, -0.8], dtype=torch.float64)
    mu = 0.5
    cfg = BarrierRecoveryConfig(root_xtol=1e-14)

    x, _ = recover_barrier_primal(prob, v, lam, gamma, mu, cfg)
    grad = prob.constraint.matvec(x) - b
    expected = float(grad @ direction)

    h = 1e-6
    phi_plus = barrier_dual_objective(prob, v, lam + h * direction, b, gamma, mu, config=cfg)
    phi_minus = barrier_dual_objective(prob, v, lam - h * direction, b, gamma, mu, config=cfg)
    finite_diff = (phi_plus - phi_minus) / (2.0 * h)
    assert finite_diff == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_barrier_recovery_rejects_nonpositive_mu():
    """The log-barrier primitive is only defined for positive ``mu``."""
    prob, v, gamma = _problem("quadratic")
    with pytest.raises(ValueError, match="mu must be > 0"):
        recover_barrier_primal(prob, v, as_tensor([0.0, 0.0]), gamma, 0.0)
