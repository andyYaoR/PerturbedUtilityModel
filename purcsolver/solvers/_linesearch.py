r"""
Armijo backtracking line search on the convex dual objective ``phi(lambda)``.

The semismooth Newton direction ``d`` solves ``(H + eps I) d = -r`` with ``r =
grad phi``, so ``grad phi . d = r . d = -r^T (H+eps I)^{-1} r < 0`` -- a descent
direction.  Backtracking ``t in {1, beta, beta^2, ...}`` until the sufficient
decrease condition

    phi(lambda + t d) <= phi(lambda) + c1 * t * (r . d)

holds globalizes the (only locally convergent) Newton iteration.

The objective is evaluated as a function of the step length ``t`` so the caller
can supply a closure that updates ``eta`` incrementally -- ``eta(t) = eta0 + t *
(A^T d / ell)`` -- avoiding a fresh ``A^T`` matvec on every backtracking step.
Because ``phi`` is ``C^1`` and convex even where the active set changes, a
sufficiently small step always succeeds, so a failure signals numerical trouble
rather than a non-descent direction.
"""

from __future__ import annotations

from typing import Callable, Tuple


def armijo_backtracking_t(
    phi_of_t: Callable[[float], float],
    phi0: float,
    directional_derivative: float,
    *,
    c1: float,
    beta: float,
    max_steps: int,
) -> Tuple[float, float, bool]:
    """
    Backtrack on the step length ``t`` until Armijo sufficient decrease holds.

    Args:
        phi_of_t: The objective as a function of step length, ``t -> phi(lambda +
            t d)`` (``phi_of_t(0) == phi0``).
        phi0: Cached value ``phi(lambda)`` at the current iterate.
        directional_derivative: ``grad phi . d`` (must be ``< 0``).
        c1: Armijo parameter in ``(0, 1/2)``.
        beta: Backtracking shrink factor in ``(0, 1)``.
        max_steps: Maximum number of halvings.

    Returns:
        ``(t, phi_new, success)``: the accepted step length, the new objective
        value, and whether sufficient decrease was achieved.

    """
    t = 1.0
    phi_trial = phi0
    for _ in range(max_steps):
        phi_trial = phi_of_t(t)
        if phi_trial <= phi0 + c1 * t * directional_derivative:
            return t, phi_trial, True
        t *= beta
    return t, phi_trial, False
