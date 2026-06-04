r"""
Armijo backtracking line search on the convex dual objective ``phi(lambda)``.

The semismooth Newton direction ``d`` solves ``(H + eps I) d = -r`` with ``r =
grad phi``, so ``grad phi . d = r . d = -r^T (H+eps I)^{-1} r < 0`` -- a descent
direction.  Backtracking ``t in {1, beta, beta^2, ...}`` until the sufficient
decrease condition

    phi(lambda + t d) <= phi(lambda) + c1 * t * (r . d)

holds globalizes the (only locally convergent) Newton iteration.  Because ``phi``
is ``C^1`` and convex even where the active set changes, a sufficiently small
step always succeeds, so a failure signals numerical trouble rather than a
non-descent direction.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from ..utils.typing import ArrayLike


def armijo_backtracking(
    phi: Callable[[ArrayLike], float],
    phi0: float,
    lam: ArrayLike,
    direction: ArrayLike,
    directional_derivative: float,
    *,
    c1: float,
    beta: float,
    max_steps: int,
) -> Tuple[ArrayLike, float, float, bool]:
    """
    Backtrack along ``direction`` until Armijo sufficient decrease holds.

    Args:
        phi: The objective ``phi(lambda)`` to minimize.
        phi0: Cached value ``phi(lam)`` at the current iterate.
        lam: Current iterate ``lambda``.
        direction: Search direction ``d`` (a descent direction).
        directional_derivative: ``grad phi . d`` (must be ``< 0``).
        c1: Armijo parameter in ``(0, 1/2)``.
        beta: Backtracking shrink factor in ``(0, 1)``.
        max_steps: Maximum number of halvings.

    Returns:
        ``(lam_new, t, phi_new, success)``: the accepted iterate, step size,
        new objective value, and whether sufficient decrease was achieved.

    """
    t = 1.0
    lam = np.asarray(lam, dtype=float)
    direction = np.asarray(direction, dtype=float)
    for _ in range(max_steps):
        lam_trial = lam + t * direction
        phi_trial = phi(lam_trial)
        if phi_trial <= phi0 + c1 * t * directional_derivative:
            return lam_trial, t, phi_trial, True
        t *= beta
    # Return the last (smallest-step) trial; caller decides how to handle failure.
    return lam_trial, t, phi_trial, False
