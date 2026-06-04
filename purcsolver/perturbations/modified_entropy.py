r"""
Modified-entropy perturbation ``h(xi) = (1+xi) log(1+xi) - xi``.

A strictly-convex kernel on ``xi > -1`` (so the whole nonnegative box is in its
domain) with an elementary inverse:

    h'(xi)  = log(1 + xi)
    h''(xi) = 1 / (1 + xi) > 0
    xi*(eta) = clip(exp(eta) - 1, lo, hi)

Unlike Shannon entropy it is finite and smooth at ``xi = 0`` (``h(0)=0``,
``h'(0)=0``), which makes it a convenient regularizer when the lower bound is an
attainable interior-feeling corner rather than a singularity.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..utils.typing import ArrayLike
from . import register_perturbation
from .base import SeparablePerturbation

_EPS = 1e-300


@register_perturbation("modified_entropy")
class ModifiedEntropyPerturbation(SeparablePerturbation):
    """Kernel ``h(xi) = (1+xi) log(1+xi) - xi`` (parameter-free)."""

    has_closed_form_recovery = True
    cvxpy_expressible = True
    default_domain = (0.0, 1.0)

    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``(1+xi) log(1+xi) - xi``."""
        del params
        xi = np.asarray(xi, dtype=float)
        one_plus = np.maximum(1.0 + xi, _EPS)
        return one_plus * np.log(one_plus) - xi

    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h'(xi) = log(1 + xi)``."""
        del params
        xi = np.asarray(xi, dtype=float)
        return np.log(np.maximum(1.0 + xi, _EPS))

    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h''(xi) = 1 / (1 + xi)``."""
        del params
        xi = np.asarray(xi, dtype=float)
        return 1.0 / np.maximum(1.0 + xi, _EPS)

    def primal_recovery(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Recover ``xi*(eta) = clip(exp(eta) - 1, lo, hi)``."""
        del params
        eta = np.asarray(eta, dtype=float)
        return self._clip_interior(np.exp(eta) - 1.0, lo, hi)

    def inv_hess_weight(self, xi_star: ArrayLike, params: Any) -> ArrayLike:
        """Return ``1 / h'' = 1 + xi*``."""
        del params
        return 1.0 + np.asarray(xi_star, dtype=float)

    def cvxpy_h(self, x: Any, params: Any) -> Any:
        """Return ``(1+xi) log(1+xi) - xi = -entr(1+xi) - xi``."""
        del params
        import cvxpy as cp

        return -cp.entr(1.0 + x) - x
