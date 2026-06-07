"""
Tests for the fixed orthonormal sieve basis (``SieveBasis``).

These are pure-algebra tests (no forward solver): the change of basis ``gamma = T c``
must (i) reduce to the identity in the monomial case, (ii) round-trip, (iii) actually
orthonormalize the shape subspace (so the working Gram is the identity while the
monomial shape Gram is Hilbert-ill-conditioned), and (iv) carry the Bernstein
convexity polyhedron over exactly as ``(M T) c >= -1`` with a feasible, idempotent
projection.  The ``nonneg`` projection is rejected in a non-monomial basis.
"""

from __future__ import annotations

import numpy as np
import pytest

from purc.estimators.debiased_fy import GammaProjection, SieveBasis
from purc.estimators.debiased_fy.projection import project_bernstein
from purc.static_purc.perturbations._bernstein import bernstein_matrix, is_convex
from purc.static_purc.utils.torch_compat import to_numpy


@pytest.mark.parametrize("L", [3, 5, 7])
def test_monomial_is_identity(L):
    b = SieveBasis.monomial(L)
    D = L - 2
    assert b.is_monomial
    np.testing.assert_array_equal(b.T, np.eye(D))
    np.testing.assert_array_equal(b.Tinv, np.eye(D))


@pytest.mark.parametrize("L", [3, 5, 7])
def test_orthonormal_roundtrip(L):
    b = SieveBasis.orthonormal(L)
    D = L - 2
    np.testing.assert_allclose(b.T @ b.Tinv, np.eye(D), atol=1e-9)
    rng = np.random.default_rng(L)
    gamma = rng.normal(size=D)
    c = to_numpy(b.from_monomial(gamma))
    np.testing.assert_allclose(to_numpy(b.to_monomial(c)), gamma, atol=1e-9)


@pytest.mark.parametrize("L", [5, 7])
def test_orthonormal_decorrelates_shape_gram(L):
    """
    The monomial shape Gram is Hilbert-ill-conditioned; the c-coordinates are
    L^2([0,1])-orthonormal (working Gram = identity).
    """
    b = SieveBasis.orthonormal(L)
    degs = np.arange(3, L + 1, dtype=float)
    gram = 1.0 / (degs[:, None] + degs[None, :] + 1.0)  # monomial shape Gram (Hilbert)
    assert np.linalg.cond(gram) > 1e3
    # c -> monomial-h coeff map is R^{-1} = Lambda^{-1} T; its Gram is the identity.
    rinv = np.diag(1.0 / degs) @ b.T
    np.testing.assert_allclose(rinv.T @ gram @ rinv, np.eye(L - 2), atol=1e-9)


@pytest.mark.parametrize("L", [4, 5, 6])
def test_convexity_pullback_matches(L):
    """is_convex(T c) holds iff the pulled-back polyhedron (M T) c >= -1 holds."""
    b = SieveBasis.orthonormal(L)
    M = bernstein_matrix(L - 2)
    MT = b.constraint_matrix(M)
    rng = np.random.default_rng(L)
    for _ in range(200):
        c = rng.normal(scale=0.5, size=L - 2)
        gamma = to_numpy(b.to_monomial(c))
        assert is_convex(gamma) == bool(np.all(MT @ c >= -1.0 - 1e-12))


@pytest.mark.parametrize("L", [4, 5])
def test_project_bernstein_in_basis_feasible_idempotent(L):
    b = SieveBasis.orthonormal(L)
    rng = np.random.default_rng(L + 1)
    c0 = rng.normal(scale=2.0, size=L - 2)  # likely infeasible
    c1 = to_numpy(project_bernstein(c0, basis=b))
    assert is_convex(to_numpy(b.to_monomial(c1)))  # projected point is convex
    c2 = to_numpy(project_bernstein(c1, basis=b))
    np.testing.assert_allclose(c2, c1, atol=1e-7)  # idempotent on feasible


def test_nonneg_in_orthonormal_basis_rejected():
    with pytest.raises(ValueError):
        GammaProjection("nonneg", basis=SieveBasis.orthonormal(5))
    # nonneg with the (default) monomial basis is fine:
    GammaProjection("nonneg", basis=SieveBasis.monomial(5))


def test_orthonormal_large_L_guarded():
    """A too-large L makes the Hilbert Cholesky/cond unusable -> a clear error."""
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        SieveBasis.orthonormal(40)
