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

import math
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch

from ..utils.torch_compat import as_tensor
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
        grad_finite_lo: Lower end of the open interval on which ``h'`` is finite.
            A *finite* value marks an endpoint where the kernel is essentially
            smooth (Legendre type): ``h'(xi) -> -inf`` as ``xi`` approaches it
            from above (e.g. Shannon entropy at ``0``).  Defaults to ``-inf``
            (gradient finite everywhere below ``grad_finite_hi``).
        grad_finite_hi: Upper end of that interval; a finite value marks
            ``h'(xi) -> +inf`` as ``xi`` approaches it from below (e.g. logit
            entropy at ``1``).  Defaults to ``+inf``.

    """

    has_closed_form_recovery: bool = False
    cvxpy_expressible: bool = False
    default_domain: Tuple[float, float] = (0.0, 1.0)
    grad_finite_lo: float = -math.inf
    grad_finite_hi: float = math.inf

    def admits_primal_interior(self, lo: ArrayLike, hi: ArrayLike) -> bool:
        r"""
        Whether a primal interior-point/log-barrier method is well posed on the box.

        This is a *provable precondition*, not a heuristic: a primal barrier
        augments stationarity with ``-mu/(x-lo) + mu/(hi-x)`` and steps ``x`` in
        the primal, so it requires ``h'`` to be **finite on the closed box**
        ``[lo, hi]``.  That holds iff the box lies strictly inside the
        finite-gradient interval ``(grad_finite_lo, grad_finite_hi)``.

        It is **false** for Legendre-type kernels whose gradient diverges at a
        box face -- Shannon entropy (``h'(0+) = -inf``) and logit entropy
        (``h'(0+) = -inf``, ``h'(1-) = +inf``).  For those the box face is
        provably never active *and* a primal barrier there is ill posed, so the
        problem must be solved in the **dual**, where the conjugate ``h*`` is
        globally smooth and the recovery ``x = h'^{-1}(eta)`` stays bounded.

        Args:
            lo: Lower box bounds (any broadcastable shape).
            hi: Upper box bounds (any broadcastable shape).

        Returns:
            ``True`` iff ``grad_finite_lo < min(lo)`` and ``max(hi) < grad_finite_hi``.

        """
        lo_min = float(as_tensor(lo).min())
        hi_max = float(as_tensor(hi).max())
        return self.grad_finite_lo < lo_min and hi_max < self.grad_finite_hi

    @abstractmethod
    def h(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Return the kernel value at ``xi`` for shape parameters ``gamma``."""

    @abstractmethod
    def hprime(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """First derivative ``h'(xi; gamma)`` (strictly increasing in ``xi``)."""

    @abstractmethod
    def hsecond(self, xi: ArrayLike, params: Any) -> ArrayLike:
        """Second derivative ``h''(xi; gamma) > 0`` on the interior."""

    def conj_box(
        self,
        eta: ArrayLike,
        lo: ArrayLike,
        hi: ArrayLike,
        params: Any,
    ) -> ArrayLike:
        """
        Box-restricted convex conjugate ``h*(eta) = max_{lo<=xi<=hi}(eta*xi - h(xi))``.

        This is the per-coordinate term of the dual objective ``phi(lambda) =
        b^T lambda + sum_i ell_i h*(eta_i)``.  The default evaluates it from
        :meth:`primal_recovery` and :meth:`h` (``eta*xi* - h(xi*)``), which is
        exact for any box; subclasses with a cheaper closed form may override.

        Args:
            eta: Reduced utilities, any broadcastable shape.
            lo: Lower box bounds.
            hi: Upper box bounds.
            params: Perturbation parameters ``gamma``.

        Returns:
            The box-restricted conjugate values, same shape as ``eta``.

        """
        eta = as_tensor(eta)
        xi_star, _ = self.primal_recovery(eta, lo, hi, params)
        return eta * xi_star - self.h(xi_star, params)

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

    def cvxpy_h(self, x: Any, params: Any) -> Any:
        """
        Return the elementwise CVXPY expression for ``h(x)`` (the oracle hook).

        Only meaningful when :attr:`cvxpy_expressible` is ``True``.  Used by the
        CVXPY reference oracle to build the equivalent convex program; ``x`` is a
        ``cvxpy.Variable``.

        Args:
            x: A ``cvxpy.Variable`` of shape ``(N,)``.
            params: Perturbation parameters ``gamma``.

        Raises:
            NotImplementedError: If the kernel is not DCP-expressible.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a CVXPY expression; "
            "use the scipy oracle instead."
        )

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
        xi = as_tensor(xi)
        lo = as_tensor(lo)
        hi = as_tensor(hi)
        clipped = torch.minimum(torch.maximum(xi, lo), hi)
        interior = (clipped > lo) & (clipped < hi)
        return clipped, interior
