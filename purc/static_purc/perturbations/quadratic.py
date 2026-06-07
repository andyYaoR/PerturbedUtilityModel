r"""
Quadratic perturbation ``h(xi) = 1/2 xi^2``.

The simplest strictly-convex kernel: ``h'(xi) = xi``, ``h''(xi) = 1``.  Its
box-restricted primal recovery is a plain projection ``xi*(eta) = clip(eta,
lo, hi)`` (per-coordinate "sparsemax on a box"), making it the ideal kernel for
validating the perturbation-agnostic solver/backend/oracle machinery before the
polynomial sieve's root-find is introduced.

The induced program ``min_x 1/2 ||x||^2_ell - v^T x  s.t.  Ax=b, l<=x<=u`` is a
box-and-equality-constrained least squares -- exactly DCP-expressible, so CVXPY
is a direct oracle.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

from ..utils.torch_compat import as_tensor
from ..utils.typing import ArrayLike
from . import register_perturbation
from .base import SeparablePerturbation


@register_perturbation("quadratic")
class QuadraticPerturbation(SeparablePerturbation):
    """Quadratic kernel ``h(xi) = xi^2 / 2`` (parameter-free)."""

    has_closed_form_recovery = True
    cvxpy_expressible = True
    barrier_kernel_code = 0

    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``xi^2 / 2``."""
        del params
        return 0.5 * as_tensor(xi) ** 2

    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h'(xi) = xi``."""
        del params
        return as_tensor(xi)

    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h''(xi) = 1``."""
        del params
        return torch.ones_like(as_tensor(xi))

    def primal_recovery(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Recover ``xi*(eta) = clip(eta, lo, hi)``."""
        del params
        return self._clip_interior(as_tensor(eta), lo, hi)

    def inv_hess_weight(self, xi_star: ArrayLike, params: Any) -> ArrayLike:
        """Return the constant Newton weight ``1 / h'' = 1``."""
        del params
        return torch.ones_like(as_tensor(xi_star))

    def cvxpy_h(self, x: Any, params: Any) -> Any:
        """Return the CVXPY expression ``xi^2 / 2``."""
        del params
        import cvxpy as cp

        return 0.5 * cp.square(x)
