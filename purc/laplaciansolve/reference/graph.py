"""
Graph / sparse-matrix utilities for the reference solver.

Ports ``lap`` (``graphOps.jl``), ``forceLap`` (``solverInterface.jl``) and
``flipIndex`` (``graphUtils.jl``) from ``Laplacians.jl``, adapted to SciPy CSC
matrices and 0-based indexing.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..logging import get_logger

_logger = get_logger(__name__)


def to_csc(a) -> sparse.csc_matrix:
    """
    Return *a* as a float64 CSC matrix with sorted indices.

    Args:
        a: Any SciPy sparse matrix or dense array convertible to CSC.

    Returns:
        The matrix as a :class:`scipy.sparse.csc_matrix` of dtype float64.

    """
    csc = sparse.csc_matrix(a, dtype=np.float64)
    csc.sort_indices()
    return csc


def column_sums(a: sparse.spmatrix) -> np.ndarray:
    """
    Return the column sums of *a* as a dense 1-D array.

    Args:
        a: A sparse matrix.

    Returns:
        Length-``n`` array of column sums.

    """
    return np.asarray(a.sum(axis=0)).ravel()


def lap(a) -> sparse.csc_matrix:
    """
    Build the combinatorial Laplacian ``L = diag(colsums(a)) - a``.

    Args:
        a: A nonnegative, symmetric adjacency matrix with zero diagonal.

    Returns:
        The Laplacian as a CSC matrix.

    """
    csc = to_csc(a)
    deg = column_sums(csc)
    return to_csc(sparse.diags(deg) - csc)


def force_lap(a) -> sparse.csc_matrix:
    """
    Coerce *a* into a Laplacian, warning if it does not look like an adjacency.

    Mirrors ``forceLap``: negative entries are taken in absolute value and the
    diagonal stripped; a present diagonal is stripped; otherwise *a* is used
    as-is.  The Laplacian ``diag(colsums) - a`` is then returned.

    Args:
        a: A matrix that should represent an adjacency.

    Returns:
        The Laplacian as a CSC matrix.

    """
    csc = to_csc(a)
    if csc.nnz and csc.min() < 0:
        _logger.warning(
            "force_lap: input has negative entries; using abs() and stripping diagonal."
        )
        csc = abs(csc)
        csc.setdiag(0)
        csc.eliminate_zeros()
    elif np.abs(csc.diagonal()).sum() > 0:
        _logger.warning("force_lap: input has a nonzero diagonal; stripping it.")
        csc.setdiag(0)
        csc.eliminate_zeros()
    deg = column_sums(csc)
    return to_csc(sparse.diags(deg) - csc)


def flip_index(a: sparse.csc_matrix) -> np.ndarray:
    """
    Map each CSC nonzero to the CSC position of its transpose ``(j, i) <-> (i, j)``.

    For a structurally symmetric matrix this is the reverse-edge map used by the
    elimination to keep the two stored copies of each edge in sync (port of
    ``flipIndex``).

    Args:
        a: A structurally symmetric CSC matrix (every ``(i, j)`` has ``(j, i)``).

    Returns:
        Integer array ``flip`` of length ``nnz`` such that the nonzero at CSC
        position ``p`` (row ``i``, column ``j``) has its twin ``(j, i)`` at CSC
        position ``flip[p]``.

    Raises:
        ValueError: If the sparsity pattern is not symmetric.

    """
    a = to_csc(a)
    n = a.shape[0]
    nnz = a.nnz
    rows = a.indices
    # Column index of every CSC nonzero (column-major run-length expansion).
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(a.indptr))

    # Position of each (row, col) pair, then look up the transposed pair.
    pos = {(int(r), int(c)): p for p, (r, c) in enumerate(zip(rows, cols))}
    flip = np.empty(nnz, dtype=np.int64)
    for p in range(nnz):
        twin = pos.get((int(cols[p]), int(rows[p])))
        if twin is None:
            raise ValueError(
                "flip_index requires a structurally symmetric matrix; "
                f"entry ({rows[p]}, {cols[p]}) has no transpose."
            )
        flip[p] = twin
    return flip


def validate_adjacency(a: sparse.csc_matrix, *, check_symmetry: bool = True) -> None:
    """
    Validate that *a* is a usable adjacency matrix.

    Args:
        a: Candidate adjacency matrix (CSC).
        check_symmetry: If ``True``, also verify numerical symmetry.

    Raises:
        ValueError: If *a* is not square, has negative weights, has a nonzero
            diagonal, or (when requested) is not symmetric.

    """
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {a.shape}")
    if a.nnz and a.min() < 0:
        raise ValueError("adjacency must have nonnegative edge weights")
    if np.abs(a.diagonal()).sum() > 0:
        raise ValueError("adjacency must have a zero diagonal (no self-loops)")
    if check_symmetry:
        diff = abs(a - a.T)
        if diff.nnz and diff.max() > 1e-12 * max(1.0, abs(a).max()):
            raise ValueError("adjacency must be symmetric")
