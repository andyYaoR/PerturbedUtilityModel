"""
Falling-factorial U-statistics for the debiased Fenchel--Young loss (torch).

For a link traversed ``n`` times out of ``D`` i.i.d. trips with link probability
``p = x*_{(i,j),b}``, the plug-in ``(n/D)^l`` is a *biased* estimator of ``p^l``.
The degree-``l`` falling-factorial U-statistic

    U_l = n^{(l)} / D^{(l)},   with  n^{(l)} = n (n-1) ... (n - l + 1),

is **unbiased**: ``E[U_l] = p^l``.  It is defined only when ``D >= l``.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ...static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor


def falling_factorial(n, deg: int) -> torch.Tensor:
    """
    Elementwise falling factorial ``n^{(deg)} = n (n-1) ... (n - deg + 1)`` (float64).

    Equals ``0`` wherever ``n < deg`` and ``1`` for ``deg == 0``.

    Args:
        n: Nonnegative counts (tensor or array), any shape.
        deg: Degree (``>= 0``).

    Returns:
        ``n^{(deg)}`` as a float64 tensor, same shape as ``n``.

    """
    n = as_tensor(n).to(DEFAULT_DTYPE)
    out = torch.ones_like(n)
    for j in range(deg):
        out = out * (n - j)
    if deg > 0:
        out = torch.where(n >= deg, out, torch.zeros_like(out))
    return out


def u_statistics(n_counts, D, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-OD, per-link U-statistics ``U_l = n^{(l)} / D^{(l)}`` for ``l = 0..L``.

    Args:
        n_counts: Link counts ``n_{(i,j),b}``, shape ``[B, N]``.
        D: Trips per OD pair ``D_b``, shape ``[B]``.
        L: Highest sieve degree.

    Returns:
        ``(U, valid)`` where ``U`` has shape ``[B, N, L + 1]`` (the unbiased
        estimator of ``(x*)^l`` in ``U[..., l]``) and ``valid`` has shape
        ``[B, L + 1]`` with ``valid[b, l] = (D_b >= l)``.

    """
    n_counts = as_tensor(n_counts).to(DEFAULT_DTYPE)
    D = as_tensor(D).to(DEFAULT_DTYPE).reshape(-1)
    B, N = n_counts.shape
    U = torch.zeros((B, N, L + 1), dtype=DEFAULT_DTYPE)
    valid = torch.zeros((B, L + 1), dtype=torch.bool)
    for deg in range(L + 1):
        Dd = falling_factorial(D, deg)  # [B]
        ok = Dd > 0
        valid[:, deg] = ok
        num = falling_factorial(n_counts, deg)  # [B, N]
        denom = torch.where(ok, Dd, torch.ones_like(Dd)).reshape(B, 1)
        U[:, :, deg] = torch.where(ok.reshape(B, 1), num / denom, torch.zeros_like(num))
    return U, valid
