r"""
Entropy perturbations (closed-form recovery).

Two classic strictly-convex kernels whose inverse ``h'^{-1}`` is elementary:

* **Shannon** ``h(xi) = xi log xi`` -- ``h'(xi) = 1 + log xi``,
  ``h''(xi) = 1/xi``, so ``xi*(eta) = clip(exp(eta - 1), lo, hi)``.  This is the
  kernel behind the recursive-logit / entropy-regularized choice.

* **Binary / logit** ``h(xi) = xi log xi + (1-xi) log(1-xi)`` on ``[0, 1]`` --
  ``h'(xi) = log(xi / (1-xi))`` (the logit), ``h''(xi) = 1/(xi(1-xi))``, so
  ``xi*(eta) = sigma(eta) = 1/(1+e^{-eta})``.  This is the canonical perturbed
  utility -> logit map: the recovery lands strictly inside ``(0,1)`` on the unit
  box, never saturating.

``scipy.special.xlogy`` evaluates ``xi log xi`` with the correct ``0 log 0 = 0``
limit, so the conjugate is well-defined at saturated corners.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.special import expit, xlogy

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

    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``xi log xi`` (with the ``0 log 0 = 0`` limit)."""
        del params
        xi = np.asarray(xi, dtype=float)
        return xlogy(xi, xi)

    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h'(xi) = 1 + log xi``."""
        del params
        xi = np.asarray(xi, dtype=float)
        return 1.0 + np.log(np.maximum(xi, _EPS))

    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h''(xi) = 1 / xi``."""
        del params
        xi = np.asarray(xi, dtype=float)
        return 1.0 / np.maximum(xi, _EPS)

    def primal_recovery(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Recover ``xi*(eta) = clip(exp(eta - 1), lo, hi)``."""
        del params
        eta = np.asarray(eta, dtype=float)
        return self._clip_interior(np.exp(eta - 1.0), lo, hi)

    def inv_hess_weight(self, xi_star: ArrayLike, params: Any) -> ArrayLike:
        """Return ``1 / h'' = xi*`` (cheaper than ``1 / hsecond``)."""
        del params
        return np.asarray(xi_star, dtype=float)

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

    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``xi log xi + (1-xi) log(1-xi)`` (with corner limits = 0)."""
        del params
        xi = np.asarray(xi, dtype=float)
        return xlogy(xi, xi) + xlogy(1.0 - xi, 1.0 - xi)

    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h'(xi) = log(xi / (1-xi))`` (the logit)."""
        del params
        xi = np.asarray(xi, dtype=float)
        return np.log(np.maximum(xi, _EPS)) - np.log(np.maximum(1.0 - xi, _EPS))

    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return ``h''(xi) = 1 / (xi (1-xi))``."""
        del params
        xi = np.asarray(xi, dtype=float)
        return 1.0 / np.maximum(xi * (1.0 - xi), _EPS)

    def primal_recovery(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Recover ``xi*(eta) = clip(sigma(eta), lo, hi)``."""
        del params
        eta = np.asarray(eta, dtype=float)
        return self._clip_interior(expit(eta), lo, hi)

    def inv_hess_weight(self, xi_star: ArrayLike, params: Any) -> ArrayLike:
        """Return ``1 / h'' = xi* (1 - xi*)``."""
        del params
        xi = np.asarray(xi_star, dtype=float)
        return xi * (1.0 - xi)

    def cvxpy_h(self, x: Any, params: Any) -> Any:
        """Return ``xi log xi + (1-xi) log(1-xi) = -entr(xi) - entr(1-xi)``."""
        del params
        import cvxpy as cp

        return -cp.entr(x) - cp.entr(1.0 - x)
