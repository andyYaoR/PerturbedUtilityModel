r"""
Abstract base class for separable perturbations.

A separable perturbation is
    ``F(x; gamma) = sum_i ell_i h(x_i; gamma)``
with ``h`` strictly convex on the box ``[lo, hi]`` (so ``h'' > 0`` on the
interior).  Everything the semismooth Newton solver needs is expressed
per-coordinate and vectorized over coordinates (and a leading batch dimension):

  * ``h, hprime, hsecond`` -- the kernel and its derivatives,
  * ``conj`` -- the per-coordinate convex conjugate ``h*`` (for the dual
    objective and the duality gap),
  * ``primal_recovery`` -- ``xi*(eta) = argmax_{xi in [lo,hi]} (eta*xi - h(xi))``,
    the dual-to-primal map, returning the ``interior_mask`` so the solver's
    active set and the inverse-Hessian weights come from a single source of
    truth (no corner-vs-mask race), and
  * ``inv_hess_weight`` -- ``1 / h''(xi*)``, the Newton edge weight on the
    active set.

Because ``h'`` is strictly increasing, ``xi*`` is the unique root of
``h'(xi) = eta`` clipped to ``[lo, hi]``; concrete subclasses provide either a
closed form or a bracketed root-find (see ``perturbations/_rootfind.py`` and the
symbolic ``perturbations/compiler.py``, added in M3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

from ..utils.typing import ArrayLike


class SeparablePerturbation(ABC):
    """
    Interface for a strictly-convex separable perturbation kernel ``h``.

    Subclasses set :attr:`has_closed_form_recovery` (whether ``primal_recovery``
    avoids a runtime root-find) and :attr:`cvxpy_expressible` (whether the
    convex program is DCP-expressible for the CVXPY oracle).

    Attributes:
        has_closed_form_recovery: ``True`` if ``xi*(eta)`` is closed form.
        cvxpy_expressible: ``True`` if ``F`` can be written in CVXPY/DCP form.
        default_domain: Default per-coordinate box ``(lo, hi)``.

    """

    has_closed_form_recovery: bool = False
    cvxpy_expressible: bool = False
    default_domain: Tuple[float, float] = (0.0, 1.0)

    @abstractmethod
    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return the kernel value at ``xi`` for shape parameters ``gamma``."""

    @abstractmethod
    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """First derivative ``h'(xi; gamma)`` (strictly increasing in ``xi``)."""

    @abstractmethod
    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Second derivative ``h''(xi; gamma) > 0`` on the interior."""

    @abstractmethod
    def conj(self, eta: ArrayLike, params: Any) -> ArrayLike:
        """Per-coordinate convex conjugate ``h*(eta) = max_xi (eta*xi - h(xi))``."""

    @abstractmethod
    def primal_recovery(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Recover the primal optimum ``xi*(eta)`` on the box ``[lo, hi]``.

        Args:
            eta: Reduced utilities, any broadcastable shape.
            lo: Lower box bounds (broadcastable to ``eta``).
            hi: Upper box bounds (broadcastable to ``eta``).
            params: Perturbation parameters ``gamma``.

        Returns:
            A tuple ``(xi_star, interior_mask)`` where ``interior_mask`` is the
            boolean array ``lo < xi_star < hi`` (the active set indicator).

        """

    def inv_hess_weight(self, xi_star: ArrayLike, params: Any) -> ArrayLike:
        """
        Newton edge weight ``D = 1 / h''(xi*)`` on the active set.

        Subclasses with a cheaper closed form should override.  Callers zero this
        outside the active set using the ``interior_mask`` from
        :meth:`primal_recovery`.

        Args:
            xi_star: Recovered primal values.
            params: Perturbation parameters ``gamma``.

        Returns:
            The per-coordinate inverse-Hessian weights.

        """
        return 1.0 / self.hsecond(xi_star, params)

    def gamma_feasible(self, params: Any) -> bool:
        """
        Whether ``params`` keeps ``h`` strictly convex on the open box.

        The default accepts everything; subclasses with a shape constraint (e.g.
        the polynomial sieve's Bernstein condition ``M gamma >= -1``) override.

        Args:
            params: Perturbation parameters ``gamma``.

        Returns:
            ``True`` if ``h'' > 0`` on the interior for these parameters.

        """
        del params
        return True

    @staticmethod
    def _clip_interior(xi: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        Clip ``xi`` to ``[lo, hi]`` and return the strict-interior mask.

        Args:
            xi: Unclipped values.
            lo: Lower bounds.
            hi: Upper bounds.

        Returns:
            ``(clipped, interior_mask)``.

        """
        clipped = np.clip(xi, lo, hi)
        interior = (clipped > lo) & (clipped < hi)
        return clipped, interior
