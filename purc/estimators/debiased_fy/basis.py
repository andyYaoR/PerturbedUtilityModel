r"""
Sieve parametrization basis: monomial (power series) vs a fixed orthonormal basis.

The polynomial sieve perturbation is stored in the *monomial* basis inside the
forward solver: ``h(xi) = 1/2 xi^2 + sum_{l=3}^L (gamma_l / l) xi^l`` with shape
coefficients ``gamma = (gamma_3, ..., gamma_L)`` (length ``D = L-2``).  On the data
support the monomials ``xi^l`` are badly conditioned (their Gram matrix on ``[0,1]``
is Hilbert-type), which is the classic ill-conditioning of power-series sieves.

This module lets the *estimator* parametrize the shape in a better-conditioned,
**fixed** orthonormal basis ``c`` with ``gamma = T c`` for a constant invertible
``T`` (size ``D x D``).  The forward solver still receives monomial ``gamma``; ``T``
lives only at the estimator boundary, so:

* **Debiasing is preserved exactly.**  ``T`` is a fixed linear map, so the
  falling-factorial U-statistics and the loss are untouched; the gamma-score just
  pulls back by ``T^T`` and ``E[score] = 0`` at ``theta_0`` still holds by linearity.
* **Convexity stays a polyhedron.**  ``{M gamma >= -1}`` becomes ``{(M T) c >= -1}``.
* The estimate is reported back in monomial ``gamma = T c`` (delta method for the
  variance, since the map is linear).

Construction of the orthonormal ``T`` (Lebesgue weight on ``[0,1]``): orthonormalize
the *shape* monomials ``{xi^3, ..., xi^L}`` **within that subspace** -- off-the-shelf
Legendre/Chebyshev would inject degree ``<= 2`` terms and corrupt the fixed
``1/2 xi^2`` baseline.  With the shape Gram ``G_{lm} = 1/(l+m+1)`` (``l,m in 3..L``)
and ``G = R^T R`` (Cholesky, ``R`` upper), the ``L^2``-orthonormal shape coefficients
``c`` map to monomial ``gamma`` by ``gamma_l = l a_l`` (since the monomial coeff of
``xi^l`` in ``h`` is ``gamma_l / l``) with ``a = R^{-1} c``, i.e.

    T = Lambda R^{-1},   Lambda = diag(l)_{l=3..L},   T^{-1} = R Lambda^{-1}.

A data-adaptive (empirical-``x*`` measure) variant is a natural extension but is left
out here: it would make ``T`` depend on ``theta`` and require sample-splitting to keep
the debiasing unbiased.  This module is intentionally *fixed*-basis only.

Honest scope: the orthonormal basis removes the basis-induced conditioning and
improves robustness/reporting, but the projected damped-Newton step is already
affine-invariant, so it does not manufacture identification of intrinsically weak
(support-driven) high-degree coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor, to_numpy

_MAX_COND_T = 1e10


@dataclass(frozen=True)
class SieveBasis:
    """
    A fixed linear parametrization of the sieve shape coefficients.

    Attributes:
        name: ``"monomial"`` (``T = I``) or ``"orthonormal"``.
        T: The ``[D, D]`` map from basis coefficients ``c`` to monomial ``gamma``
            (``gamma = T c``), ``D = L - 2``.
        Tinv: The inverse map ``c = Tinv gamma``.
        L: Highest sieve degree.

    """

    name: str
    T: np.ndarray
    Tinv: np.ndarray
    L: int

    @classmethod
    def monomial(cls, L: int) -> "SieveBasis":
        """The identity parametrization ``c = gamma`` (power-series basis)."""
        D = max(L - 2, 0)
        eye = np.eye(D, dtype=float)
        return cls(name="monomial", T=eye, Tinv=eye.copy(), L=L)

    @classmethod
    def orthonormal(cls, L: int, *, on: str = "h") -> "SieveBasis":
        """
        The ``L^2([0,1])``-orthonormal shape basis (a fixed, Legendre-style basis).

        Args:
            L: Highest sieve degree.
            on: Which object to orthonormalize; only ``"h"`` is implemented (a
                ``"hpp"`` variant orthonormalizing ``h''`` is a documented future hook).

        Returns:
            The basis with ``T = Lambda R^{-1}`` (see module docstring).

        Raises:
            NotImplementedError: If ``on`` is not ``"h"``.
            ValueError: If the resulting ``T`` is too ill-conditioned (large ``L``).

        """
        if on != "h":
            raise NotImplementedError("only on='h' is implemented (hpp is a future hook)")
        D = max(L - 2, 0)
        if D == 0:
            eye = np.eye(0, dtype=float)
            return cls(name="orthonormal", T=eye, Tinv=eye.copy(), L=L)
        degs = np.arange(3, L + 1, dtype=float)  # shape degrees 3..L
        gram = 1.0 / (degs[:, None] + degs[None, :] + 1.0)  # Hilbert-type shape Gram
        r = np.linalg.cholesky(gram).T  # upper R with G = R^T R
        rinv = np.linalg.inv(r)
        lam = np.diag(degs)
        T = lam @ rinv  # gamma = Lambda R^{-1} c
        Tinv = r @ np.diag(1.0 / degs)  # c = R Lambda^{-1} gamma
        cond = float(np.linalg.cond(T))
        if not np.isfinite(cond) or cond > _MAX_COND_T:
            raise ValueError(
                f"orthonormal basis ill-conditioned (cond(T)={cond:.2e}); L={L} too large"
            )
        return cls(name="orthonormal", T=T, Tinv=Tinv, L=L)

    @property
    def is_monomial(self) -> bool:
        """Whether this is the identity (power-series) parametrization."""
        return self.name == "monomial"

    def to_monomial(self, c):
        """Map basis coefficients ``c`` to monomial ``gamma = T c`` (torch)."""
        c = as_tensor(c).to(DEFAULT_DTYPE).reshape(-1)
        if self.is_monomial:
            return c
        return as_tensor(self.T).to(DEFAULT_DTYPE) @ c

    def from_monomial(self, gamma):
        """Map monomial ``gamma`` to basis coefficients ``c = Tinv gamma`` (torch)."""
        gamma = as_tensor(gamma).to(DEFAULT_DTYPE).reshape(-1)
        if self.is_monomial:
            return gamma
        return as_tensor(self.Tinv).to(DEFAULT_DTYPE) @ gamma

    def constraint_matrix(self, M: np.ndarray) -> np.ndarray:
        """Pull a monomial-space constraint matrix ``M`` back to ``c``-space: ``M T``."""
        return np.asarray(M, dtype=float) @ self.T

    def pushforward_var(self, var_block):
        """Delta-method map of a ``c``-space covariance block to monomial ``gamma``: ``T V T^T``."""
        T = self.T
        V = to_numpy(var_block)
        return as_tensor(T @ V @ T.T).to(DEFAULT_DTYPE)
