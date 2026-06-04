r"""
Vectorized root-finding for strictly-monotone scalar equations.

The dual-to-primal map requires ``xi*(eta) = g^{-1}(eta)`` clipped to ``[lo, hi]``
where ``g = h'`` is strictly increasing (because ``h'' > 0``).  The interior root
is therefore unique and bracketed by ``[lo, hi]``, which lets us use a
safeguarded Newton iteration ("rtsafe"): a Newton step when it stays inside the
current bracket and reduces the residual, a bisection step otherwise.  This is
robust even where ``g'`` is small near a corner, needs only ``g`` and ``g'`` (no
third derivative, unlike Halley), and is fully vectorized over all coordinates
and any leading batch dimension -- the reference for the native kernel in v0.3.0.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from ..utils.typing import ArrayLike

_TINY = 1e-300


def solve_monotone(
    g: Callable[[ArrayLike], ArrayLike],
    gprime: Callable[[ArrayLike], ArrayLike],
    target: ArrayLike,
    lo: ArrayLike,
    hi: ArrayLike,
    *,
    max_iter: int = 80,
    xtol: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve ``g(xi) = target`` on ``[lo, hi]`` for strictly increasing ``g``.

    Coordinates whose target falls outside ``[g(lo), g(hi)]`` saturate to the
    corresponding bound; the rest are solved by safeguarded Newton.

    Args:
        g: The strictly increasing function (e.g. ``h'``), vectorized.
        gprime: Its derivative (e.g. ``h''``), vectorized and positive.
        target: Right-hand side ``eta``, any shape broadcastable with the bounds.
        lo: Lower bounds (broadcastable to ``target``).
        hi: Upper bounds (broadcastable to ``target``).
        max_iter: Maximum safeguarded-Newton iterations.
        xtol: Absolute step tolerance for convergence.

    Returns:
        ``(xi_star, interior_mask)`` with ``xi_star`` clipped to ``[lo, hi]`` and
        ``interior_mask`` the boolean array ``lo < xi_star < hi``.

    """
    target, lo, hi = np.broadcast_arrays(
        np.asarray(target, dtype=float),
        np.asarray(lo, dtype=float),
        np.asarray(hi, dtype=float),
    )
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    g_lo = g(lo)
    g_hi = g(hi)

    below = target <= g_lo
    above = target >= g_hi
    interior = ~below & ~above

    # Bracket [a, b] and an initial linear-interpolation guess for interior coords.
    a = lo.copy()
    b = hi.copy()
    denom = np.where(interior, g_hi - g_lo, 1.0)
    frac = np.where(interior, (target - g_lo) / np.where(denom == 0, _TINY, denom), 0.0)
    x = np.where(below, lo, np.where(above, hi, lo + frac * (hi - lo)))

    for _ in range(max_iter):
        fx = g(x) - target
        # g increasing: f > 0 means the root is to the left -> tighten the upper end.
        b = np.where(interior & (fx > 0), x, b)
        a = np.where(interior & (fx <= 0), x, a)

        dfx = gprime(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_newton = x - fx / dfx
        out_of_bracket = ~np.isfinite(x_newton) | (x_newton <= a) | (x_newton >= b)
        x_next = np.where(out_of_bracket, 0.5 * (a + b), x_newton)

        step = np.where(interior, x_next - x, 0.0)
        x = np.where(interior, x_next, x)
        if np.all(np.abs(step) <= xtol * (1.0 + np.abs(x))):
            break

    x = np.clip(x, lo, hi)
    interior = (x > lo) & (x < hi)
    return x, interior
