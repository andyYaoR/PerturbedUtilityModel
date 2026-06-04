"""
Sparse-matrix helpers for constraint handling and Newton-system assembly.

Centralizes the small, easy-to-get-wrong conversions (dense/sparse/torch -> CSR
float), the row-grouping used to reason about the multiplier nullspace, and the
``b in range(A)`` feasibility test that tells the SSN solver up front whether the
residual can ever reach zero.
"""

from __future__ import annotations

from typing import Tuple

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
