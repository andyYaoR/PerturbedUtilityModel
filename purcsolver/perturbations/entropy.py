r"""
Entropy perturbations (closed-form recovery).

Two classic strictly-convex kernels whose inverse ``h'^{-1}`` is elementary:

* **Shannon** ``h(xi) = xi log xi`` -- ``h'(xi) = 1 + log xi``,
  ``h''(xi) = 1/xi``, so ``xi*(eta) = clip(exp(eta - 1), lo, hi)``.

* **Binary / logit** ``h(xi) = xi log xi + (1-xi) log(1-xi)`` on ``[0, 1]`` --
  ``h'(xi) = log(xi / (1-xi))`` (the logit), ``h''(xi) = 1/(xi(1-xi))``, so
  ``xi*(eta) = sigma(eta)``.  The canonical perturbed-utility -> logit map.

``torch.special.xlogy`` carries the ``0 log 0 = 0`` limit, so the conjugate is
well-defined at saturated corners.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

from ..utils.torch_compat import as_tensor
from ..utils.typing import ArrayLike
from . import register_perturbation
from .base import SeparablePerturbation

# Floor used only inside derivative evaluations to avoid log(0) / divide-by-zero
# when a coordinate sits exactly on a bound; never changes recovered values.
_EPS = 1e-300


@register_perturbation("entropy")
@register_perturbation("shannon_entropy")
class EntropyPerturbation(SeparablePerturbation):
    """Shannon entropy kernel ``h(xi) = xi log xi`` (parameter-free)."""

    has_closed_form_recovery = True
    cvxpy_expressible = True
    default_domain = (0.0, 1.0)
    # Legendre type at the lower bound: h'(xi) = 1 + log xi -> -inf as xi -> 0+,
    # so a primal box barrier at x = 0 is ill posed; solve in the dual.  h'(1) = 1
    # is finite, so the gradient stays finite for all xi > 0 (grad_finite_hi = +inf).
    grad_finite_lo = 0.0
    barrier_kernel_code = 1

    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``xi log xi`` (with the ``0 log 0 = 0`` limit)."""
        del params
        xi = as_tensor(xi)
        return torch.special.xlogy(xi, xi)

    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h'(xi) = 1 + log xi``."""
        del params
        xi = as_tensor(xi)
        return 1.0 + torch.log(xi.clamp_min(_EPS))

    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h''(xi) = 1 / xi``."""
        del params
        xi = as_tensor(xi)
        return 1.0 / xi.clamp_min(_EPS)

    def primal_recovery(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Recover ``xi*(eta) = clip(exp(eta - 1), lo, hi)``."""
        del params
        eta = as_tensor(eta)
        return self._clip_interior(torch.exp(eta - 1.0), lo, hi)

    def inv_hess_weight(self, xi_star: ArrayLike, params: Any) -> ArrayLike:
        """Return ``1 / h'' = xi*`` (cheaper than ``1 / hsecond``)."""
        del params
        return as_tensor(xi_star)

    def cvxpy_h(self, x: Any, params: Any) -> Any:
        """Return ``xi log xi = -entr(xi)`` as a CVXPY expression."""
        del params
        import cvxpy as cp

        return -cp.entr(x)


@register_perturbation("logit_entropy")
@register_perturbation("binary_entropy")
class LogitEntropyPerturbation(SeparablePerturbation):
    """Binary entropy ``h(xi) = xi log xi + (1-xi) log(1-xi)`` on ``[0, 1]``."""

    has_closed_form_recovery = True
    cvxpy_expressible = True
    default_domain = (0.0, 1.0)
    # Legendre type at BOTH bounds: h'(xi) = log(xi/(1-xi)) -> -inf at 0+ and
    # +inf at 1-.  The solution is provably interior and a primal box barrier is
    # ill posed at either face; solve in the dual.
    grad_finite_lo = 0.0
    grad_finite_hi = 1.0
    barrier_kernel_code = 2

    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``xi log xi + (1-xi) log(1-xi)`` (with corner limits = 0)."""
        del params
        xi = as_tensor(xi)
        return torch.special.xlogy(xi, xi) + torch.special.xlogy(1.0 - xi, 1.0 - xi)

    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h'(xi) = log(xi / (1-xi))`` (the logit)."""
        del params
        xi = as_tensor(xi)
        return torch.log(xi.clamp_min(_EPS)) - torch.log((1.0 - xi).clamp_min(_EPS))

    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h''(xi) = 1 / (xi (1-xi))``."""
        del params
        xi = as_tensor(xi)
        return 1.0 / (xi * (1.0 - xi)).clamp_min(_EPS)

    def primal_recovery(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Recover ``xi*(eta) = clip(sigma(eta), lo, hi)``."""
        del params
        eta = as_tensor(eta)
        return self._clip_interior(torch.special.expit(eta), lo, hi)

    def inv_hess_weight(self, xi_star: ArrayLike, params: Any) -> ArrayLike:
        """Return ``1 / h'' = xi* (1 - xi*)``."""
        del params
        xi = as_tensor(xi_star)
        return xi * (1.0 - xi)

    def cvxpy_h(self, x: Any, params: Any) -> Any:
        """Return ``xi log xi + (1-xi) log(1-xi) = -entr(xi) - entr(1-xi)``."""
        del params
        import cvxpy as cp

        return -cp.entr(x) - cp.entr(1.0 - x)
