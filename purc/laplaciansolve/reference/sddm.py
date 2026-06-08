"""
SDDM <-> Laplacian transformation.

An SDDM matrix ``M`` (symmetric, positive diagonal, nonpositive off-diagonals,
diagonally dominant) is solved by embedding it into a Laplacian on one extra
"ground" vertex that carries each row's dominance excess, then solving the
augmented Laplacian system and dropping the ground entry.  Ports
``sddmWrapLap`` / ``adjValAndExcess`` / ``extendMatrix`` from ``Laplacians.jl``.

For the PURC system ``M = H + eps*I`` (``H`` a weighted Laplacian) every row's
excess equals ``eps``, so every vertex connects to ground with weight ``eps``.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy import sparse

from .graph import to_csc


def adj_val_and_excess(
    m,
) -> Tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """
    Split an SDDM matrix into adjacency, diagonal, and dominance excess.

    Args:
        m: A symmetric SDDM matrix.

    Returns:
        A triple ``(a, diag, excess)`` where ``a`` is the off-diagonal adjacency
        (``a_ij = -m_ij >= 0``), ``diag`` is the diagonal of ``m``, and
        ``excess`` is the per-row dominance ``excess_i = sum_j m_ij >= 0``.

    Raises:
        ValueError: If ``m`` has a positive off-diagonal entry (not an M-matrix).

    """
    m = to_csc(m)
    diag = m.diagonal().astype(np.float64)
    off = to_csc(m - sparse.diags(diag))
    if off.nnz and off.max() > 1e-12 * max(1.0, float(np.abs(diag).max())):
        raise ValueError("SDDM matrix must have nonpositive off-diagonal entries")
    a = to_csc(-off)
    a.eliminate_zeros()
    excess = np.asarray(m.sum(axis=1)).ravel().astype(np.float64)
    return a, diag, excess


def extend_matrix(a: sparse.csc_matrix, excess: np.ndarray) -> Tuple[sparse.csc_matrix, bool]:
    """
    Append a ground vertex carrying the dominance excess.

    Args:
        a: Off-diagonal adjacency of the SDDM matrix.
        excess: Per-row dominance excess (length ``n``).

    Returns:
        A pair ``(a1, extended)``.  When some excess is positive, ``a1`` is the
        ``(n + 1) x (n + 1)`` augmented adjacency and ``extended`` is ``True``;
        otherwise ``a1`` is ``a`` unchanged and ``extended`` is ``False`` (the
        matrix was already a Laplacian).

    """
    n = a.shape[0]
    threshold = 1e-12 * max(1.0, float(np.abs(excess).max()) if excess.size else 1.0)
    pos = np.flatnonzero(excess > threshold)
    if pos.size == 0:
        return to_csc(a), False

    ground = n
    coo = a.tocoo()
    rows = np.concatenate([coo.row, pos, np.full(pos.size, ground)])
    cols = np.concatenate([coo.col, np.full(pos.size, ground), pos])
    vals = np.concatenate([coo.data, excess[pos], excess[pos]])
    a1 = sparse.coo_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    return to_csc(a1), True


def sddm_wrap_lap(
    lap_solver_factory: Callable[[sparse.csc_matrix], Callable[[np.ndarray], np.ndarray]],
) -> Callable[[object], Callable[[np.ndarray], np.ndarray]]:
    """
    Turn a Laplacian-solver factory into an SDDM-solver factory.

    Args:
        lap_solver_factory: Maps an adjacency matrix to a solver ``f(b) -> x`` for
            its Laplacian (e.g. :func:`~purc.laplaciansolve.reference.solver.approxchol_lap`).

    Returns:
        A function mapping an SDDM matrix to a solver ``f(b) -> x`` for that matrix.

    """

    def make_solver(m) -> Callable[[np.ndarray], np.ndarray]:
        a, _diag, excess = adj_val_and_excess(m)
        a1, extended = extend_matrix(a, excess)
        n = a.shape[0]

        if not extended:
            # Already a Laplacian: solve directly on the adjacency.
            return lap_solver_factory(a1)

        solve_aug = lap_solver_factory(a1)

        def solve(b: np.ndarray) -> np.ndarray:
            """
            Solve ``M x = b`` via the grounded augmented Laplacian.

            Args:
                b: Right-hand side of length ``n``.

            Returns:
                The solution vector of length ``n``.

            """
            b = np.asarray(b, dtype=np.float64)
            b_aug = np.concatenate([b, [-float(b.sum())]])
            x_aug = solve_aug(b_aug)
            x_aug = x_aug - x_aug[-1]
            return x_aug[:n]

        return solve

    return make_solver
