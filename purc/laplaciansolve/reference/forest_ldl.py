"""
Exact leaf-pruning LDL for forests (the acyclic fast path).

When the active subgraph is a forest the Laplacian factorizes *exactly* with no
fill-in: eliminating a leaf records a single multiplier and never alters the
remaining edges.  This is the common case at the optimum of the PURC Newton
iteration, so it gets a dedicated exact routine here.

Eliminating leaves in order is precisely the ``:deg`` ordering restricted to a
forest, so the resulting :class:`LDLinv` is consumed by the same
:func:`~purc.laplaciansolve.reference.ldl_solve.ldl_solve` as the approximate build —
only here the factorization is exact and deterministic.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np
from scipy import sparse

from .graph import to_csc
from .ldl_solve import LDLinv, ldl_solve


def _adjacency_lists(a: sparse.csc_matrix) -> List[Dict[int, float]]:
    """
    Build mutable per-vertex neighbor->weight maps from a CSC adjacency.

    Args:
        a: Symmetric adjacency matrix (CSC).

    Returns:
        A list whose ``j``-th entry maps each neighbor of ``j`` to its weight.

    """
    n = a.shape[0]
    indptr = a.indptr
    indices = a.indices
    data = a.data
    nbrs: List[Dict[int, float]] = [dict() for _ in range(n)]
    for j in range(n):
        for p in range(int(indptr[j]), int(indptr[j + 1])):
            nbrs[j][int(indices[p])] = float(data[p])
    return nbrs


def is_forest(a) -> bool:
    """
    Return whether the graph with adjacency *a* is a forest (acyclic).

    Args:
        a: Symmetric adjacency matrix.

    Returns:
        ``True`` if the graph has no cycles, else ``False``.

    """
    csc = to_csc(a)
    n = csc.shape[0]
    # Undirected edge count = nnz / 2 (symmetric, zero diagonal).
    n_edges = csc.nnz // 2
    from .components import components as _components

    n_comp, _ = _components(csc)
    return n_edges == n - n_comp


def forest_ldl(a) -> LDLinv:
    """
    Build the exact :class:`LDLinv` of ``lap(a)`` for a forest *a*.

    Args:
        a: Symmetric, nonnegative, zero-diagonal adjacency matrix that is a
            forest (CSC).

    Returns:
        The exact factorization.

    Raises:
        ValueError: If *a* contains a cycle (not a forest).

    """
    csc = to_csc(a)
    n = csc.shape[0]
    nbrs = _adjacency_lists(csc)
    deg = np.array([len(nbrs[i]) for i in range(n)], dtype=np.int64)
    eliminated = np.zeros(n, dtype=bool)

    order: List[int] = []
    parent: List[int] = []
    pweight: List[float] = []

    leaves: deque[int] = deque(int(i) for i in range(n) if deg[i] == 1)
    while leaves:
        v = leaves.popleft()
        if eliminated[v] or deg[v] != 1:
            continue
        u = next(iter(nbrs[v]))  # the unique remaining neighbor
        w = nbrs[v][u]
        order.append(v)
        parent.append(u)
        pweight.append(w)
        eliminated[v] = True
        # Detach v <-> u.
        del nbrs[u][v]
        del nbrs[v][u]
        deg[u] -= 1
        deg[v] = 0
        if deg[u] == 1 and not eliminated[u]:
            leaves.append(int(u))

    # Any surviving edge means an unbroken cycle.
    for i in range(n):
        if not eliminated[i] and deg[i] >= 1:
            raise ValueError("forest_ldl requires an acyclic graph; cycle detected")

    n_elim = len(order)
    col = np.asarray(order, dtype=np.int64)
    colptr = np.arange(n_elim + 1, dtype=np.int64)  # one entry per eliminated column
    rowval = np.asarray(parent, dtype=np.int64)
    fval = np.ones(n_elim, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    for v, w in zip(order, pweight):
        d[v] = w

    return LDLinv(col=col, colptr=colptr, rowval=rowval, fval=fval, d=d)


def forest_solve(a, b: np.ndarray, *, subtract_mean: bool = True) -> np.ndarray:
    """
    Exactly solve ``lap(a) x = b`` for a forest *a*.

    Args:
        a: Forest adjacency matrix (CSC).
        b: Right-hand side (mean-zero per connected component for consistency).
        subtract_mean: Whether to project out the global constant component.

    Returns:
        The exact solution vector.

    """
    return ldl_solve(forest_ldl(a), np.asarray(b, dtype=np.float64), subtract_mean=subtract_mean)
