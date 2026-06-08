r"""
Small dense convex QP for the trust-region subproblem (torch float64).

Solves
    min_d  1/2 d^T B d + g^T d   s.t.   lo <= d <= hi,   A d >= a,
with ``B`` symmetric positive definite (the damped-BFGS model), a box (the
``ell_inf`` trust region), and optional linear inequalities (the Bernstein
polyhedron ``M (gamma + d_gamma) >= -1`` enters as ``A d >= a``).

The method is a **primal active-set** scheme started from the feasible point
``d = 0`` (always feasible here: ``lo <= 0 <= hi`` and ``a <= 0``), following
Nocedal--Wright Alg. 16.3.  It is exact, finite, deterministic, and warm-startable
(the working set), and ports directly to the native kernel.  All inequalities are
treated uniformly as rows ``c_i^T d >= b_i``; the box contributes ``2P`` rows.

This is the reference (fallback) implementation; :func:`solve_box_linear_qp`
dispatches to the native kernel when available and falls back to this otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...static_purc.utils.native import native_available, native_core
from ...static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor, to_numpy


@dataclass
class QPResult:
    """Outcome of a box+linear QP solve."""

    d: torch.Tensor          # the minimizer, shape [P]
    active: torch.Tensor     # bool mask over the assembled rows [2P + m]
    multipliers: torch.Tensor  # KKT multipliers >= 0 over all rows [2P + m]
    iters: int               # active-set iterations
    converged: bool


def _assemble(lo, hi, A, a, P, dtype):
    """
    Stack box (lo<=d<=hi) and linear (A d>=a) into **unit-norm** rows ``C d >= b``.

    Row normalization leaves the feasible set unchanged but keeps the KKT system
    well-conditioned regardless of the raw constraint scaling (a constraint with a
    tiny normal would otherwise corrupt the step).  ``norms`` is returned so the
    multipliers can be mapped back to the original rows.
    """
    eye = torch.eye(P, dtype=dtype)
    rows = [eye, -eye]               # d >= lo ; -d >= -hi
    rhs = [lo, -hi]
    if A is not None and A.numel():
        rows.append(A)
        rhs.append(a)
    C = torch.cat(rows, dim=0)
    b = torch.cat(rhs, dim=0)
    norms = torch.linalg.norm(C, dim=1).clamp_min(1e-300)
    return C / norms.unsqueeze(1), b / norms, norms


def solve_box_linear_qp_torch(
    B, g, lo, hi, A=None, a=None, *, tol: float = 1e-11, max_iter: int | None = None
) -> QPResult:
    """
    Reference primal active-set solver (torch float64).  See module docstring.

    Args:
        B: SPD model Hessian ``[P, P]``.
        g: Linear term ``[P]``.
        lo, hi: Box bounds ``[P]`` (the trust region; require ``lo <= 0 <= hi``).
        A, a: Optional linear inequality rows ``A d >= a`` (``A`` is ``[m, P]``).
        tol: Activity / optimality tolerance.
        max_iter: Cap on active-set iterations (default ``10(P+m)+50``).

    Returns:
        A :class:`QPResult` with the minimizer and the KKT certificate.

    """
    B = as_tensor(B).to(DEFAULT_DTYPE)
    g = as_tensor(g).to(DEFAULT_DTYPE).reshape(-1)
    P = g.shape[0]
    lo = as_tensor(lo).to(DEFAULT_DTYPE).reshape(-1)
    hi = as_tensor(hi).to(DEFAULT_DTYPE).reshape(-1)
    A = None if A is None else as_tensor(A).to(DEFAULT_DTYPE).reshape(-1, P)
    a = None if a is None else as_tensor(a).to(DEFAULT_DTYPE).reshape(-1)
    C, b, row_norms = _assemble(lo, hi, A, a, P, B.dtype)
    m = C.shape[0]
    if max_iter is None:
        max_iter = 10 * (P + m) + 50

    d = torch.zeros(P, dtype=B.dtype)
    work = (C @ d - b).abs() <= tol  # working set: rows active at d=0
    iters = 0
    for iters in range(1, max_iter + 1):
        idx = torch.nonzero(work, as_tuple=False).reshape(-1)
        c = B @ d + g
        if idx.numel() == 0:
            p = -torch.linalg.solve(B, c)
            mu = torch.zeros(0, dtype=B.dtype)
        else:
            Cw = C.index_select(0, idx)            # [k, P]
            k = idx.numel()
            kkt = torch.zeros((P + k, P + k), dtype=B.dtype)
            kkt[:P, :P] = B
            kkt[:P, P:] = -Cw.T
            kkt[P:, :P] = Cw
            rhs = torch.zeros(P + k, dtype=B.dtype)
            rhs[:P] = -c
            # Unit-norm rows + the blocking test keep C_W full row rank, so the
            # KKT is nonsingular; lstsq is a defensive fallback for exact ties.
            try:
                sol = torch.linalg.solve(kkt, rhs)
            except RuntimeError:
                sol = torch.linalg.lstsq(kkt, rhs).solution
            p, mu = sol[:P], sol[P:]

        if torch.max(torch.abs(p)) <= tol:           # step ~ 0: test optimality
            if idx.numel() == 0 or bool(torch.all(mu >= -tol)):
                full_mu = torch.zeros(m, dtype=B.dtype)
                if idx.numel():
                    full_mu[idx] = mu
                full_mu = full_mu / row_norms  # map multipliers back to original rows
                return QPResult(d=d, active=work.clone(), multipliers=full_mu,
                                iters=iters, converged=True)
            # Bland's rule: drop the *lowest-index* row with a negative multiplier
            # (least-index pivoting is what guarantees no cycling on degenerate QPs).
            neg = torch.nonzero(mu < -tol, as_tuple=False).reshape(-1)
            work[idx[int(neg[0])]] = False
            continue

        Cp = C @ p                                   # ratio test over inactive rows
        blocking = (~work) & (Cp < -tol)
        alpha = torch.tensor(1.0, dtype=B.dtype)
        jblock = -1
        if bool(blocking.any()):
            bi = torch.nonzero(blocking, as_tuple=False).reshape(-1)
            ratios = (b.index_select(0, bi) - C.index_select(0, bi) @ d) / (C.index_select(0, bi) @ p)
            amin = torch.min(ratios)
            if amin < alpha:
                alpha = amin
                # Bland's rule: among rows attaining the min ratio, add the lowest index.
                tied = bi[(ratios <= amin + tol)]
                jblock = int(tied.min())
        d = d + alpha * p
        if jblock >= 0:
            work[jblock] = True

    full_mu = torch.zeros(m, dtype=B.dtype)
    return QPResult(d=d, active=work.clone(), multipliers=full_mu, iters=iters, converged=False)


def _solve_native(B, g, lo, hi, A, a, tol: float):
    """Call the native ``qp_box_linear_f64`` kernel; ``None`` if it does not converge."""
    core = native_core()
    if core is None or not hasattr(core, "qp_box_linear_f64"):
        return None
    P = int(g.shape[0])
    Bn = np.ascontiguousarray(to_numpy(B).reshape(-1), dtype=np.float64)
    gn = np.ascontiguousarray(to_numpy(g).reshape(-1), dtype=np.float64)
    lon = np.ascontiguousarray(to_numpy(lo).reshape(-1), dtype=np.float64)
    hin = np.ascontiguousarray(to_numpy(hi).reshape(-1), dtype=np.float64)
    m = 0 if A is None else int(A.shape[0])
    An = np.ascontiguousarray(to_numpy(A).reshape(-1), dtype=np.float64) if m else np.zeros(0)
    an = np.ascontiguousarray(to_numpy(a).reshape(-1), dtype=np.float64) if m else np.zeros(0)
    d = np.zeros(P, dtype=np.float64)
    info = np.zeros(2, dtype=np.float64)
    core.qp_box_linear_f64(Bn, gn, lon, hin, An, an, P, m, tol, 0, d, info)
    if info[1] != 1.0:  # native did not converge -> let the caller fall back
        return None
    return QPResult(d=torch.from_numpy(d).to(DEFAULT_DTYPE), active=torch.zeros(0, dtype=torch.bool),
                    multipliers=torch.zeros(0, dtype=DEFAULT_DTYPE), iters=int(info[0]), converged=True)


def solve_box_linear_qp(B, g, lo, hi, A=None, a=None, *, tol: float = 1e-11, max_iter=None,
                        prefer_native: bool = True) -> QPResult:
    """
    Solve the box+linear QP, dispatching to the native kernel when available.

    The native ``qp_box_linear_f64`` (in ``_static_purc_core``) reproduces
    :func:`solve_box_linear_qp_torch` to ``~1e-11``; the torch reference is the
    correctness backstop (used when the native module is absent or a solve fails to
    converge).  ``QPResult.active``/``multipliers`` are populated only by the torch
    path (the native path returns the minimizer ``d`` and ``[iters, converged]``).

    Args:
        B, g, lo, hi, A, a: As in :func:`solve_box_linear_qp_torch`.
        tol: Activity / optimality tolerance.
        max_iter: Cap on active-set iterations (torch path; native auto-sizes).
        prefer_native: Use the native kernel when available (default ``True``).

    Returns:
        A :class:`QPResult`.

    """
    if prefer_native and native_available():
        B_t = as_tensor(B).to(DEFAULT_DTYPE)
        g_t = as_tensor(g).to(DEFAULT_DTYPE).reshape(-1)
        lo_t = as_tensor(lo).to(DEFAULT_DTYPE).reshape(-1)
        hi_t = as_tensor(hi).to(DEFAULT_DTYPE).reshape(-1)
        A_t = None if A is None else as_tensor(A).to(DEFAULT_DTYPE).reshape(-1, g_t.shape[0])
        a_t = None if a is None else as_tensor(a).to(DEFAULT_DTYPE).reshape(-1)
        res = _solve_native(B_t, g_t, lo_t, hi_t, A_t, a_t, tol)
        if res is not None:
            return res
    return solve_box_linear_qp_torch(B, g, lo, hi, A=A, a=a, tol=tol, max_iter=max_iter)
