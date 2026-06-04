"""
Tests for the closed-form separable perturbations (v0.1.0).

For each registered closed-form kernel we check, on its interior:
  * derivative consistency: ``hprime``/``hsecond`` match finite differences of
    ``h``,
  * strict convexity: ``hsecond > 0``,
  * the inverse relation: for an interior recovery, ``h'(xi*) == eta``,
  * monotonicity of ``xi*(eta)`` in ``eta``,
  * box saturation: ``xi*`` clips to the bounds and ``interior_mask`` agrees,
  * the conjugate identity: ``conj_box(eta) == eta*xi* - h(xi*)`` and the
    Fenchel–Young inequality ``h(xi) + conj_box(eta) >= eta*xi``.
"""

from __future__ import annotations

import numpy as np
import pytest

from purcsolver.perturbations import PERTURBATIONS, get_perturbation

CLOSED_FORM = ["quadratic", "entropy", "logit_entropy", "modified_entropy"]


@pytest.fixture(params=CLOSED_FORM)
def kernel(request):
    return request.param, get_perturbation(request.param)


def _interior_grid(name):
    # Points strictly inside (0,1), avoiding the singular corners of entropy.
    return np.linspace(0.05, 0.95, 19)


def test_registered():
    for name in CLOSED_FORM:
        assert name in PERTURBATIONS


def test_derivatives_match_finite_difference(kernel):
    name, pert = kernel
    xi = _interior_grid(name)
    g = pert.hprime(xi, None)
    g2 = pert.hsecond(xi, None)
    eps = 1e-6
    fd1 = (pert.h(xi + eps, None) - pert.h(xi - eps, None)) / (2 * eps)
    fd2 = (pert.h(xi + eps, None) - 2 * pert.h(xi, None) + pert.h(xi - eps, None)) / eps**2
    np.testing.assert_allclose(g, fd1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(g2, fd2, rtol=1e-3, atol=1e-4)


def test_strict_convexity(kernel):
    _, pert = kernel
    xi = _interior_grid(None)
    assert np.all(pert.hsecond(xi, None) > 0)


def test_inverse_relation_interior(kernel):
    name, pert = kernel
    # eta range and box chosen so every kernel's recovery lands strictly inside
    # (modified entropy's exp(eta)-1 grows fastest, so the box is generous above).
    eta = np.linspace(-1.5, 1.5, 21)
    lo = np.full_like(eta, -2.0)
    hi = np.full_like(eta, 8.0)
    xi_star, interior = pert.primal_recovery(eta, lo, hi, None)
    assert np.all(interior)  # wide box => never saturates
    np.testing.assert_allclose(pert.hprime(xi_star, None), eta, rtol=1e-7, atol=1e-9)


def test_recovery_monotone(kernel):
    _, pert = kernel
    eta = np.linspace(-3.0, 3.0, 50)
    lo = np.full_like(eta, -10.0)
    hi = np.full_like(eta, 10.0)
    xi_star, _ = pert.primal_recovery(eta, lo, hi, None)
    assert np.all(np.diff(xi_star) >= -1e-12)


def test_box_saturation_and_mask(kernel):
    _, pert = kernel
    # A tight interior box forces saturation for every kernel: entropy/logit
    # only approach their natural [0,1] corners asymptotically, so a [0,1] box
    # would never clip them -- [0.2, 0.8] does.
    eta = np.array([-50.0, 0.0, 50.0])
    lo = np.full_like(eta, 0.2)
    hi = np.full_like(eta, 0.8)
    xi_star, interior = pert.primal_recovery(eta, lo, hi, None)
    assert xi_star[0] == pytest.approx(0.2, abs=1e-9)  # clipped to lo
    assert xi_star[2] == pytest.approx(0.8, abs=1e-9)  # clipped to hi
    assert not interior[0] and not interior[2]
    assert (xi_star >= lo - 1e-12).all() and (xi_star <= hi + 1e-12).all()


def test_conjugate_identity_and_fenchel_young(kernel):
    _, pert = kernel
    eta = np.linspace(-2.0, 2.0, 21)
    lo = np.zeros_like(eta)
    hi = np.ones_like(eta)
    xi_star, _ = pert.primal_recovery(eta, lo, hi, None)
    conj = pert.conj_box(eta, lo, hi, None)
    np.testing.assert_allclose(conj, eta * xi_star - pert.h(xi_star, None), atol=1e-12)
    # Fenchel–Young: h(xi) + h*(eta) >= eta*xi for any feasible xi.
    xi_probe = np.linspace(0.05, 0.95, 21)
    assert np.all(pert.h(xi_probe, None) + conj - eta * xi_probe >= -1e-9)


def test_inv_hess_weight_matches_reciprocal(kernel):
    _, pert = kernel
    xi = _interior_grid(None)
    np.testing.assert_allclose(
        pert.inv_hess_weight(xi, None), 1.0 / pert.hsecond(xi, None), rtol=1e-9
    )
