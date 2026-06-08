"""
Shared pytest fixtures and graph generators for the LaplacianSolve test suite.

Ensures the repo root is importable, exposes a ``tolerance`` helper (mirroring
PUM), and provides reproducible graph generators covering the regimes the solver
must handle: connected random graphs, grids (cyclic), trees/forests (acyclic),
and disconnected unions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
from scipy import sparse

# (purc.laplaciansolve is installed in-tree -- no sys.path hack needed.)


def tolerance(dtype: np.dtype = np.float64) -> Tuple[float, float]:
    """
    Return ``(atol, rtol)`` appropriate for *dtype*.

    Args:
        dtype: The floating dtype under test.

    Returns:
        An ``(atol, rtol)`` pair.

    """
    if np.dtype(dtype) == np.float32:
        return 1e-2, 1e-1
    return 1e-6, 1e-4


def symmetrize(a: np.ndarray) -> sparse.csc_matrix:
    """
    Symmetrize a dense weight matrix into a zero-diagonal CSC adjacency.

    Args:
        a: Dense square weight matrix.

    Returns:
        ``(triu(a, 1) + triu(a, 1).T)`` as CSC.

    """
    upper = np.triu(a, 1)
    return sparse.csc_matrix(upper + upper.T)


def random_connected_graph(n: int, p: float = 0.05, seed: int = 0) -> sparse.csc_matrix:
    """
    Random weighted graph made connected by adding a Hamiltonian path.

    Args:
        n: Number of vertices.
        p: Erdos-Renyi edge probability for the extra edges.
        seed: RNG seed.

    Returns:
        A connected, symmetric, positively weighted adjacency (CSC).

    """
    r = np.random.default_rng(seed)
    mask = (r.random((n, n)) < p).astype(float)
    for i in range(n - 1):  # guarantee connectivity
        mask[i, i + 1] = 1.0
    weights = r.random((n, n)) + 0.1
    return symmetrize(mask * weights)


def grid_graph(rows: int, cols: int, seed: int = 0) -> sparse.csc_matrix:
    """
    2-D weighted grid graph (rich in cycles).

    Args:
        rows: Number of grid rows.
        cols: Number of grid columns.
        seed: RNG seed for edge weights.

    Returns:
        The grid adjacency (CSC).

    """
    r = np.random.default_rng(seed)
    n = rows * cols
    ri, ci, vi = [], [], []

    def add(u: int, v: int) -> None:
        w = float(r.random() + 0.1)
        ri.extend([u, v])
        ci.extend([v, u])
        vi.extend([w, w])

    for i in range(rows):
        for j in range(cols):
            u = i * cols + j
            if j + 1 < cols:
                add(u, u + 1)
            if i + 1 < rows:
                add(u, u + cols)
    return sparse.csc_matrix((vi, (ri, ci)), shape=(n, n))


def random_tree(n: int, seed: int = 0) -> sparse.csc_matrix:
    """
    Random weighted spanning tree on ``n`` vertices (a forest).

    Args:
        n: Number of vertices.
        seed: RNG seed.

    Returns:
        The tree adjacency (CSC).

    """
    r = np.random.default_rng(seed)
    ri, ci, vi = [], [], []
    for v in range(1, n):
        u = int(r.integers(0, v))
        w = float(r.random() + 0.2)
        ri.extend([u, v])
        ci.extend([v, u])
        vi.extend([w, w])
    return sparse.csc_matrix((vi, (ri, ci)), shape=(n, n))


def disconnected_graph(seed: int = 0) -> Tuple[sparse.csc_matrix, int]:
    """
    Block-diagonal union of two triangles plus an isolated vertex.

    Args:
        seed: RNG seed for edge weights.

    Returns:
        A pair ``(adjacency, n_components)``.

    """
    r = np.random.default_rng(seed)

    def triangle() -> np.ndarray:
        w = r.random(3) + 0.3
        t = np.zeros((3, 3))
        t[0, 1] = t[1, 0] = w[0]
        t[1, 2] = t[2, 1] = w[1]
        t[0, 2] = t[2, 0] = w[2]
        return t

    block = sparse.block_diag([triangle(), triangle(), np.zeros((1, 1))], format="csc")
    return sparse.csc_matrix(block), 3


def sddm_from_adjacency(a: sparse.csc_matrix, eps: float) -> sparse.csc_matrix:
    """
    Build the PURC-style SDDM matrix ``M = lap(a) + eps * I``.

    Args:
        a: Adjacency matrix.
        eps: Positive diagonal regularizer.

    Returns:
        The SDDM matrix as CSC.

    """
    from purc.laplaciansolve.reference import lap

    n = a.shape[0]
    return sparse.csc_matrix(lap(a) + eps * sparse.eye(n))


def mean_zero(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Draw a mean-zero standard-normal vector of length ``n``.

    Args:
        rng: A NumPy generator.
        n: Vector length.

    Returns:
        A mean-centered random vector.

    """
    b = rng.standard_normal(n)
    return b - b.mean()


@pytest.fixture
def rng() -> np.random.Generator:
    """
    A seeded NumPy generator for test right-hand sides.

    Returns:
        A deterministic generator.

    """
    return np.random.default_rng(12345)
