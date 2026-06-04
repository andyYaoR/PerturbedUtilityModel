r"""
Polynomial-sieve perturbation (the paper's "semi-nonparametric" kernel).

    h(xi; gamma) = 1/2 xi^2 + sum_{l=3}^L gamma_l xi^l / l
    h'(xi)       = xi + sum_{l=3}^L gamma_l xi^{l-1}
    h''(xi)      = 1 + sum_{l=3}^L gamma_l (l-1) xi^{l-2}

The quadratic term is fixed (normalization) and ``gamma = (gamma_3, ..., gamma_L)``
are the estimated shape parameters living in the Bernstein polyhedron
``Gamma_B = {gamma : M gamma >= -1}`` that guarantees ``h'' > 0`` on ``(0,1)``
(see ``_bernstein``).  ``h'`` is then strictly increasing, so the primal recovery
``xi*(eta) = h'^{-1}(eta)`` clipped to the box is the unique bracketed root and is
computed by the vectorized safeguarded-Newton solver in ``_rootfind``.  CVXPY
cannot express a general sieve, so the independent scipy dual oracle is the ground
truth in tests.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

from ..utils.torch_compat import as_tensor, to_numpy
from ..utils.typing import ArrayLike
from . import register_perturbation
from ._bernstein import is_convex
from ._rootfind import solve_monotone
from .base import SeparablePerturbation


@register_perturbation("polynomial_sieve")
class PolynomialSievePerturbation(SeparablePerturbation):
    """
    Degree-``L`` polynomial sieve with shape parameters ``gamma_3..gamma_L``.

    The shape parameters may be supplied at construction (fixed kernel) and/or
    passed per call as ``params`` (the estimation sweep); a per-call ``params``
    overrides the stored default.

    Args:
        gamma: Default shape parameters ``(gamma_3, ..., gamma_L)``; ``None`` or
            empty means the pure quadratic kernel.

    """

    has_closed_form_recovery = False
    cvxpy_expressible = False
    default_domain = (0.0, 1.0)

    def __init__(self, gamma: ArrayLike | None = None) -> None:
        self.gamma = (
            torch.zeros(0, dtype=torch.float64) if gamma is None else as_tensor(gamma).reshape(-1)
        )

    # -- parameter handling -------------------------------------------------

    def _coeffs(self, params: Any) -> torch.Tensor:
        """
        Resolve the shape parameters for a call (per-call overrides default).

        Args:
            params: Per-call ``gamma`` or ``None``/empty to use the stored default.

        Returns:
            The shape-parameter tensor ``(gamma_3, ..., gamma_L)``.

        """
        if params is None:
            return self.gamma
        arr = as_tensor(params).reshape(-1)
        return self.gamma if arr.numel() == 0 else arr

    # -- kernel and derivatives ---------------------------------------------

    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``1/2 xi^2 + sum_l gamma_l xi^l / l``."""
        xi = as_tensor(xi)
        out = 0.5 * xi**2
        for m, g in enumerate(self._coeffs(params)):
            ll = m + 3
            out = out + g * xi**ll / ll
        return out

    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``xi + sum_l gamma_l xi^{l-1}``."""
        xi = as_tensor(xi)
        out = xi.clone()
        for m, g in enumerate(self._coeffs(params)):
            out = out + g * xi ** (m + 2)
        return out

    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``1 + sum_l gamma_l (l-1) xi^{l-2}``."""
        xi = as_tensor(xi)
        out = torch.ones_like(xi)
        for m, g in enumerate(self._coeffs(params)):
            out = out + g * (m + 2) * xi ** (m + 1)
        return out

    # -- recovery / feasibility --------------------------------------------

    def primal_recovery(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Recover ``xi*(eta)`` by inverting the monotone ``h'`` on ``[lo, hi]``."""
        coeffs = self._coeffs(params)
        return solve_monotone(
            lambda z: self.hprime(z, coeffs),
            lambda z: self.hsecond(z, coeffs),
            eta,
            lo,
            hi,
        )

    def gamma_feasible(self, params: Any) -> bool:
        """Return whether ``params`` satisfies the Bernstein condition ``M gamma >= -1``."""
        return is_convex(to_numpy(self._coeffs(params)))
