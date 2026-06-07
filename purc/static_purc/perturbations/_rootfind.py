r"""
Vectorized root-finding for strictly-monotone scalar equations (torch).

The dual-to-primal map requires ``xi*(eta) = g^{-1}(eta)`` clipped to ``[lo, hi]``
where ``g = h'`` is strictly increasing (because ``h'' > 0``).  The interior root
is therefore unique and bracketed by ``[lo, hi]``, which lets us use a
safeguarded Newton iteration ("rtsafe"): a Newton step when it stays inside the
current bracket, a bisection step otherwise.  This is robust even where ``g'`` is
small near a corner, needs only ``g`` and ``g'`` (no third derivative, unlike
Halley), and is fully vectorized over all coordinates and any leading batch
dimension on torch tensors -- the reference for the native kernel in v0.3.0.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch

from ..utils.torch_compat import as_tensor
from ..utils.typing import ArrayLike

_TINY = 1e-300


def solve_monotone(
    g: Callable[[torch.Tensor], torch.Tensor],
    gprime: Callable[[torch.Tensor], torch.Tensor],
    target: ArrayLike,
    lo: ArrayLike,
    hi: ArrayLike,
    *,
    max_iter: int = 80,
    xtol: float = 1e-14,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve ``g(xi) = target`` on ``[lo, hi]`` for strictly increasing ``g``.

    Coordinates whose target falls outside ``[g(lo), g(hi)]`` saturate to the
    corresponding bound; the rest are solved by safeguarded Newton.

    Args:
        g: The strictly increasing function (e.g. ``h'``), vectorized.
        gprime: Its derivative (e.g. ``h''``), vectorized and positive.
        target: Right-hand side ``eta`` (broadcastable with the bounds).
        lo: Lower bounds (broadcastable to ``target``).
        hi: Upper bounds (broadcastable to ``target``).
        max_iter: Maximum safeguarded-Newton iterations.
        xtol: Absolute step tolerance for convergence.

    Returns:
        ``(xi_star, interior_mask)`` with ``xi_star`` clipped to ``[lo, hi]`` and
        ``interior_mask`` the boolean tensor ``lo < xi_star < hi``.

    """
    target, lo, hi = torch.broadcast_tensors(as_tensor(target), as_tensor(lo), as_tensor(hi))
    lo = lo.clone()
    hi = hi.clone()
    g_lo = g(lo)
    g_hi = g(hi)

    below = target <= g_lo
    above = target >= g_hi
    interior = ~below & ~above

    a = lo.clone()
    b = hi.clone()
    denom = torch.where(interior, g_hi - g_lo, torch.ones_like(target))
    denom = torch.where(denom == 0, torch.full_like(denom, _TINY), denom)
    frac = torch.where(interior, (target - g_lo) / denom, torch.zeros_like(target))
    x = torch.where(below, lo, torch.where(above, hi, lo + frac * (hi - lo)))

    for _ in range(max_iter):
        fx = g(x) - target
        b = torch.where(interior & (fx > 0), x, b)
        a = torch.where(interior & (fx <= 0), x, a)

        dfx = gprime(x)
        x_newton = x - fx / dfx
        out_of_bracket = (~torch.isfinite(x_newton)) | (x_newton <= a) | (x_newton >= b)
        x_next = torch.where(out_of_bracket, 0.5 * (a + b), x_newton)

        step = torch.where(interior, x_next - x, torch.zeros_like(x))
        x = torch.where(interior, x_next, x)
        if bool(torch.all(torch.abs(step) <= xtol * (1.0 + torch.abs(x)))):
            break

    x = torch.minimum(torch.maximum(x, lo), hi)
    interior = (x > lo) & (x < hi)
    return x, interior
