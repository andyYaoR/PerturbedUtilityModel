"""
Sparse-matrix helpers for constraint handling and Newton-system assembly.

Centralizes the small, easy-to-get-wrong conversions (dense/sparse/torch -> CSR
float), the row-grouping used to reason about the multiplier nullspace, and the
``b in range(A)`` feasibility test that tells the SSN solver up front whether the
residual can ever reach zero.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr


def to_csr(matrix) -> sp.csr_matrix:
    """
    Coerce a dense array / scipy sparse / CPU torch tensor to ``csr_matrix`` of
    floats.

    Args:
        matrix: A 2-D array-like (numpy, scipy sparse, or torch CPU tensor).

    Returns:
        The matrix as a float64 ``scipy.sparse.csr_matrix``.

    Raises:
        ValueError: If a dense input is not 2-dimensional.

    """
    if sp.issparse(matrix):
        return matrix.tocsr().astype(float)
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {arr.shape}")
    return sp.csr_matrix(arr)


def row_components(A: sp.csr_matrix) -> Tuple[int, np.ndarray]:
    """
    Group equality rows by the variables they share.

    Two rows are connected when some column is nonzero in both; the connected
    components partition the rows.  For a node-arc incidence matrix these are
    exactly the graph's connected components, and the per-component indicator
    vectors span the left-nullspace of ``A`` (the gauge freedom in ``lambda``).

    Args:
        A: The ``(k, N)`` constraint matrix (CSR).

    Returns:
        ``(n_components, labels)`` where ``labels`` has shape ``(k,)``.

    """
    # Row adjacency: rows i, j connected iff (A A^T)_{ij} != 0.
    row_adj = (A @ A.T).tocsr()
    n_comp, labels = connected_components(row_adj, directed=False)
    return n_comp, labels


def detect_incidence(A: sp.csr_matrix) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Detect whether ``A`` is a node-arc incidence matrix and return its edges.

    ``A`` (``k`` nodes x ``N`` edges) is an incidence matrix iff every column has
    exactly two nonzeros equal to ``+1`` and ``-1``.  When so, the per-edge
    Newton system ``A diag(D) A^T + eps I`` is a weighted graph Laplacian and can
    be routed to LaplacianSolve's specialized ``PURCLaplacianSolver`` (fused
    native assembly + solve, forest fast path).

    Args:
        A: The ``(k, N)`` constraint matrix (CSR).

    Returns:
        ``(is_incidence, edges)`` where ``edges`` is an ``(N, 2)`` int64 array of
        the two endpoint nodes per edge (in column order), or ``None`` if ``A`` is
        not an incidence matrix.

    """
    Acsc = A.tocsc()
    indptr, indices, data = Acsc.indptr, Acsc.indices, Acsc.data
    n_edges = Acsc.shape[1]
    edges = np.empty((n_edges, 2), dtype=np.int64)
    for j in range(n_edges):
        seg = slice(indptr[j], indptr[j + 1])
        rows = indices[seg]
        vals = data[seg]
        if rows.size != 2:
            return False, None
        order = np.argsort(vals)
        if not (np.isclose(vals[order[0]], -1.0) and np.isclose(vals[order[1]], 1.0)):
            return False, None
        edges[j] = rows
    return True, edges


def in_range(A: sp.csr_matrix, b: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Test whether ``b`` lies in the column space of ``A`` (i.e. ``A x = b`` is
    solvable ignoring the box).

    A necessary condition for the SSN residual ``r = b - A x_hat`` to reach zero.
    Uses LSQR to find ``min_x ||A x - b||`` and checks the relative residual.

    Args:
        A: The ``(k, N)`` constraint matrix (CSR).
        b: Right-hand side, shape ``(k,)``.
        tol: Relative residual tolerance.

    Returns:
        ``True`` if ``b`` is (numerically) in ``range(A)``.

    """
    b = np.asarray(b, dtype=float).ravel()
    nb = np.linalg.norm(b)
    if nb == 0.0:
        return True
    # LSQR returns the min-norm least-squares solution; r1norm is ||Ax-b||.
    out = lsqr(A, b, atol=1e-12, btol=1e-12)
    residual = out[3]  # r1norm
    return residual <= tol * nb
