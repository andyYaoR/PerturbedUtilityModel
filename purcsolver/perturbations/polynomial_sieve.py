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

import numpy as np
import torch

from ..utils.native import native_available, native_core
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

    # -- kernel and derivatives (Horner) ------------------------------------

    @staticmethod
    def _horner(xi: torch.Tensor, coeffs_low_to_high: list) -> torch.Tensor:
        """
        Evaluate ``sum_k c_k xi^k`` by Horner's method (one pass).

        Args:
            xi: Evaluation points.
            coeffs_low_to_high: Coefficients ``[c_0, c_1, ..., c_d]``.

        Returns:
            The polynomial value at ``xi``.

        """
        out = torch.full_like(xi, float(coeffs_low_to_high[-1]))
        for c in reversed(coeffs_low_to_high[:-1]):
            out = out * xi + c
        return out

    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``1/2 xi^2 + sum_l gamma_l xi^l / l``."""
        xi = as_tensor(xi)
        g = self._coeffs(params)
        # c_2 = 1/2, c_l = gamma_l / l for l = 3..L.
        coeffs = [0.0, 0.0, 0.5] + [g[m] / (m + 3) for m in range(g.numel())]
        return self._horner(xi, coeffs)

    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``xi + sum_l gamma_l xi^{l-1}``."""
        xi = as_tensor(xi)
        g = self._coeffs(params)
        # c_1 = 1, c_{l-1} = gamma_l for l = 3..L.
        coeffs = [0.0, 1.0] + [g[m] for m in range(g.numel())]
        return self._horner(xi, coeffs)

    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``1 + sum_l gamma_l (l-1) xi^{l-2}``."""
        xi = as_tensor(xi)
        g = self._coeffs(params)
        # c_0 = 1, c_{l-2} = gamma_l (l-1) for l = 3..L.
        coeffs = [1.0] + [g[m] * (m + 2) for m in range(g.numel())]
        return self._horner(xi, coeffs)

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
        if native_available():
            return self._recover_native(eta, lo, hi, coeffs)
        return solve_monotone(
            lambda z: self.hprime(z, coeffs),
            lambda z: self.hsecond(z, coeffs),
            eta,
            lo,
            hi,
        )

    def _recover_native(
        self, eta: ArrayLike, lo: ArrayLike, hi: ArrayLike, coeffs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recover ``xi*(eta)`` via the native polynomial-inversion kernel.

        Args:
            eta: Reduced utilities.
            lo: Lower bounds (broadcastable to ``eta``).
            hi: Upper bounds (broadcastable to ``eta``).
            coeffs: The sieve shape parameters ``gamma``.

        Returns:
            ``(xi_star, interior_mask)`` as torch tensors on ``eta``'s device.

        """
        eta_t, lo_t, hi_t = torch.broadcast_tensors(as_tensor(eta), as_tensor(lo), as_tensor(hi))
        shape = tuple(eta_t.shape)
        # h' coefficients (low to high): c_0 = 0, c_1 = 1, c_{l-1} = gamma_l.
        hp_coeffs = np.concatenate([[0.0, 1.0], to_numpy(coeffs)]).astype(np.float64)
        et = np.ascontiguousarray(to_numpy(eta_t).ravel())
        lon = np.ascontiguousarray(to_numpy(lo_t).ravel())
        hin = np.ascontiguousarray(to_numpy(hi_t).ravel())
        xi = np.empty(et.size, dtype=np.float64)
        interior = np.empty(et.size, dtype=np.uint8)
        native_core().recovery_poly_f64(hp_coeffs, et, lon, hin, xi, interior)
        return (
            as_tensor(xi.reshape(shape), device=eta_t.device),
            as_tensor(interior.reshape(shape).astype(bool), dtype=torch.bool, device=eta_t.device),
        )

    def gamma_feasible(self, params: Any) -> bool:
        """Return whether ``params`` satisfies the Bernstein condition ``M gamma >= -1``."""
        return is_convex(to_numpy(self._coeffs(params)))
