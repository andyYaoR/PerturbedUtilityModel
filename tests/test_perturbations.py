"""
Tests for the closed-form separable perturbations (torch-native).

For each registered closed-form kernel we check, on its interior:
  * derivative consistency: ``hprime``/``hsecond`` match finite differences of
    ``h``,
  * strict convexity: ``hsecond > 0``,
  * the inverse relation: for an interior recovery, ``h'(xi*) == eta``,
  * monotonicity of ``xi*(eta)`` in ``eta``,
  * box saturation: ``xi*`` clips to the bounds and ``interior_mask`` agrees,
  * the conjugate identity and the Fenchel-Young inequality.

Kernels are torch-native; results are bridged to NumPy with ``to_numpy`` for the
NumPy-based assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from purcsolver.perturbations import PERTURBATIONS, get_perturbation
from purcsolver.utils.torch_compat import to_numpy

CLOSED_FORM = ["quadratic", "entropy", "logit_entropy", "modified_entropy"]


@pytest.fixture(params=CLOSED_FORM)
def kernel(request):
    return request.param, get_perturbation(request.param)


def _interior_grid():
    return torch.linspace(0.05, 0.95, 19, dtype=torch.float64)


def test_registered():
    for name in CLOSED_FORM:
        assert name in PERTURBATIONS


def test_derivatives_match_finite_difference(kernel):
    _, pert = kernel
    xi = _interior_grid()
    g = to_numpy(pert.hprime(xi, None))
    g2 = to_numpy(pert.hsecond(xi, None))
    eps = 1e-6
    fd1 = to_numpy(pert.h(xi + eps, None) - pert.h(xi - eps, None)) / (2 * eps)
    fd2 = to_numpy(pert.h(xi + eps, None) - 2 * pert.h(xi, None) + pert.h(xi - eps, None)) / eps**2
    np.testing.assert_allclose(g, fd1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(g2, fd2, rtol=1e-3, atol=1e-4)


def test_strict_convexity(kernel):
    _, pert = kernel
    assert bool((pert.hsecond(_interior_grid(), None) > 0).all())


def test_inverse_relation_interior(kernel):
    _, pert = kernel
    eta = torch.linspace(-1.5, 1.5, 21, dtype=torch.float64)
    lo = torch.full_like(eta, -2.0)
    hi = torch.full_like(eta, 8.0)
    xi_star, interior = pert.primal_recovery(eta, lo, hi, None)
    assert bool(interior.all())
    np.testing.assert_allclose(
        to_numpy(pert.hprime(xi_star, None)), to_numpy(eta), rtol=1e-7, atol=1e-9
    )


def test_recovery_monotone(kernel):
    _, pert = kernel
    eta = torch.linspace(-3.0, 3.0, 50, dtype=torch.float64)
    lo = torch.full_like(eta, -10.0)
    hi = torch.full_like(eta, 10.0)
    xi_star, _ = pert.primal_recovery(eta, lo, hi, None)
    assert bool((torch.diff(xi_star) >= -1e-12).all())


def test_box_saturation_and_mask(kernel):
    _, pert = kernel
    eta = torch.tensor([-50.0, 0.0, 50.0], dtype=torch.float64)
    lo = torch.full_like(eta, 0.2)
    hi = torch.full_like(eta, 0.8)
    xi_star, interior = pert.primal_recovery(eta, lo, hi, None)
    xi_star = to_numpy(xi_star)
    assert xi_star[0] == pytest.approx(0.2, abs=1e-9)
    assert xi_star[2] == pytest.approx(0.8, abs=1e-9)
    assert not bool(interior[0]) and not bool(interior[2])
    assert (xi_star >= 0.2 - 1e-12).all() and (xi_star <= 0.8 + 1e-12).all()


def test_conjugate_identity_and_fenchel_young(kernel):
    _, pert = kernel
    eta = torch.linspace(-2.0, 2.0, 21, dtype=torch.float64)
    lo = torch.zeros_like(eta)
    hi = torch.ones_like(eta)
    xi_star, _ = pert.primal_recovery(eta, lo, hi, None)
    conj = pert.conj_box(eta, lo, hi, None)
    np.testing.assert_allclose(
        to_numpy(conj), to_numpy(eta * xi_star - pert.h(xi_star, None)), atol=1e-12
    )
    # Fenchel-Young: h(xi) + h*(eta) >= eta*xi for any feasible xi.
    xi_probe = torch.linspace(0.05, 0.95, 21, dtype=torch.float64)
    assert bool((pert.h(xi_probe, None) + conj - eta * xi_probe >= -1e-9).all())


def test_inv_hess_weight_matches_reciprocal(kernel):
    _, pert = kernel
    xi = _interior_grid()
    np.testing.assert_allclose(
        to_numpy(pert.inv_hess_weight(xi, None)),
        to_numpy(1.0 / pert.hsecond(xi, None)),
        rtol=1e-9,
    )
