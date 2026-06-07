r"""
Bernstein-coefficient convexity certificate for the polynomial sieve.

The sieve kernel ``h(xi) = 1/2 xi^2 + sum_{l=3}^L gamma_l xi^l / l`` is strictly
convex on ``(0, 1)`` iff ``h''(xi) = 1 + sum_{l=3}^L gamma_l (l-1) xi^{l-2} > 0``
there.  Verifying positivity of a polynomial is, in general, hard; the standard
*sufficient* condition is that its Bernstein coefficients on ``[0, 1]`` are
nonnegative (Bernstein coefficients bound the polynomial, so ``c_k >= 0 =>
h'' >= 0``).  Writing ``h''`` in the power basis with ``a_0 = 1`` and
``a_{l-2} = gamma_l (l-1)``, the degree-``d = L-2`` Bernstein coefficients are

    c_k = sum_{j=0}^k (C(k,j) / C(d,j)) a_j = 1 + (M gamma)_k,

so the inner-approximating feasible region is the polyhedron ``Gamma_B = {gamma :
M gamma >= -1}``.  ``M`` (shape ``(L-1, L-2)``) is fixed and precomputable.
"""

from __future__ import annotations

from math import comb

import numpy as np


def bernstein_matrix(num_gamma: int) -> np.ndarray:
    """
    Build the Bernstein constraint matrix ``M`` for ``num_gamma`` sieve weights.

    Args:
        num_gamma: Number of shape parameters ``gamma_3, ..., gamma_L`` (``L-2``).

    Returns:
        The ``(L-1, L-2)`` matrix ``M`` such that the Bernstein coefficients of
        ``h''`` are ``c = 1 + M @ gamma``.

    """
    g = num_gamma  # L - 2
    d = g  # degree of h'' is L - 2
    M = np.zeros((d + 1, g), dtype=float)
    for k in range(d + 1):
        for m in range(g):
            j = m + 1  # power of xi for gamma_{m+3}
            if j <= k:
                # a_j = gamma_{m+3} * (j+1) = gamma * (m+2)
                M[k, m] = comb(k, j) / comb(d, j) * (m + 2)
    return M


def bernstein_coeffs(gamma: np.ndarray) -> np.ndarray:
    """
    Bernstein coefficients ``c = 1 + M gamma`` of ``h''`` on ``[0, 1]``.

    Args:
        gamma: Shape parameters, shape ``(L-2,)``.

    Returns:
        The ``(L-1,)`` Bernstein coefficient vector.

    """
    gamma = np.asarray(gamma, dtype=float)
    if gamma.size == 0:
        return np.ones(1)
    return 1.0 + bernstein_matrix(gamma.size) @ gamma


def is_convex(gamma: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Test the Bernstein sufficient condition ``M gamma >= -1`` for convexity.

    Args:
        gamma: Shape parameters, shape ``(L-2,)``.
        tol: Slack tolerance.

    Returns:
        ``True`` if all Bernstein coefficients of ``h''`` are ``>= -tol``.

    """
    return bool(np.all(bernstein_coeffs(gamma) >= -tol))
