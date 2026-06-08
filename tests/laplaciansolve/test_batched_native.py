"""
Tests for the native batched factorization bindings.

Covers ``ForestBatchSolver`` (exact zero-fill, the acyclic route) and
``BatchedCholmodSolver`` (direct sparse Cholesky, the cyclic route): per-system
batches of B distinct matrices sharing one sparsity pattern, with ragged
right-hand-side blocks, checked against per-system dense solves and against
looping the single-matrix path.  The higher-level ``BatchedSDDMSolver`` (routing
+ array I/O) is tested separately.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from purc.laplaciansolve._loader import cholmod_core, has_cholmod, native_core
from purc.laplaciansolve.reference import lap


def _grid_adjacency(g: int) -> sp.csc_matrix:
    """Return the 4-neighbour grid-graph adjacency on g*g nodes (cyclic)."""
    rows: list[int] = []
    cols: list[int] = []

    def ix(i: int, j: int) -> int:
        return i * g + j

    for i in range(g):
        for j in range(g):
            if i + 1 < g:
                rows += [ix(i, j), ix(i + 1, j)]
                cols += [ix(i + 1, j), ix(i, j)]
            if j + 1 < g:
                rows += [ix(i, j), ix(i, j + 1)]
                cols += [ix(i, j + 1), ix(i, j)]
    n = g * g
    return sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _path_adjacency(n: int) -> sp.csc_matrix:
    """Return a path-graph adjacency on n nodes (a forest / spanning tree)."""
    rows = []
    cols = []
    for i in range(n - 1):
        rows += [i, i + 1]
        cols += [i + 1, i]
    return sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _sddm(adj: sp.csc_matrix, eps: float) -> sp.csc_matrix:
    """Build the SDDM matrix M = lap(adj) + eps*I as a sorted float64 CSC."""
    m = (lap(adj) + eps * sp.identity(adj.shape[0])).tocsc()
    m.sort_indices()
    return m


def _ragged_offsets(ks: list[int]) -> np.ndarray:
    """Return the int64 [B+1] offsets for per-system RHS-block sizes ``ks``."""
    return np.concatenate([[0], np.cumsum(ks)]).astype(np.int64)


def _per_system_values(adj, scales, epses, indptr, indices):
    """Build a [B, nnz] value block of distinct SDDM matrices on one pattern."""
    values = []
    mats = []
    for w, eps in zip(scales, epses, strict=True):
        ms = _sddm(w * adj, eps)
        assert np.array_equal(ms.indices.astype(np.int64), indices)
        assert np.array_equal(ms.indptr.astype(np.int64), indptr)
        values.append(ms.data.astype(np.float64))
        mats.append(ms)
    return np.ascontiguousarray(np.stack(values)), mats


def _worst_residual(mats, offsets, rhs, xout):
    """Return the worst relative residual over all systems and their RHS."""
    worst = 0.0
    for s, ms in enumerate(mats):
        dense = ms.toarray()
        for r in range(int(offsets[s]), int(offsets[s + 1])):
            worst = max(
                worst, np.linalg.norm(dense @ xout[r] - rhs[r]) / np.linalg.norm(rhs[r])
            )
    return worst


def test_forest_batch_per_system_matches_dense():
    """Per-system forest batch (distinct trees, ragged RHS) solves each system."""
    adj = _path_adjacency(20)
    m0 = _sddm(adj, 0.3)
    indptr = m0.indptr.astype(np.int64)
    indices = m0.indices.astype(np.int64)
    n = m0.shape[0]
    fb = native_core().ForestBatchSolver(indptr, indices, n)
    assert fb.is_forest()

    scales = [1.0, 2.5, 0.5, 1.7]
    epses = [0.3, 0.1, 0.9, 0.2]
    values, mats = _per_system_values(adj, scales, epses, indptr, indices)
    ks = [2, 1, 3, 0]  # ragged, including an empty block
    offs = _ragged_offsets(ks)
    total = int(offs[-1])
    rng = np.random.default_rng(0)
    rhs = np.ascontiguousarray(rng.standard_normal((total, n)))
    xout = np.empty((total, n))
    fb.solve_batch(values, rhs, offs, xout)
    assert _worst_residual(mats, offs, rhs, xout) < 1e-10


def test_forest_batch_shared_matches_single():
    """Shared-matrix forest batch equals looping the single-matrix solve."""
    adj = _path_adjacency(15)
    m = _sddm(adj, 0.4)
    indptr = m.indptr.astype(np.int64)
    indices = m.indices.astype(np.int64)
    n = m.shape[0]
    fb = native_core().ForestBatchSolver(indptr, indices, n)
    single = native_core().ForestSolver(indptr, indices, m.data.astype(np.float64), n)

    rng = np.random.default_rng(1)
    batch = rng.standard_normal((6, n))
    rhs = np.ascontiguousarray(batch)
    xshared = np.empty((6, n))
    fb.solve_batch_shared(m.data.astype(np.float64), rhs, xshared)
    for r in range(6):
        xref = np.zeros(n)
        single.solve(np.ascontiguousarray(batch[r]), xref, 1)
        assert np.allclose(xshared[r], xref, atol=1e-12)


@pytest.mark.skipif(not has_cholmod(), reason="CHOLMOD not built")
def test_cholmod_batch_per_system_matches_dense():
    """Per-system CHOLMOD batch (distinct cyclic SPD matrices) solves each system."""
    adj = _grid_adjacency(12)
    m0 = _sddm(adj, 0.5)
    indptr = m0.indptr.astype(np.int64)
    indices = m0.indices.astype(np.int64)
    n = m0.shape[0]
    bc = cholmod_core().BatchedCholmodSolver(indptr, indices, m0.data.astype(np.float64), n)

    scales = [1.0, 2.0, 0.5, 1.3, 0.8]
    epses = [0.5, 0.2, 0.9, 0.3, 0.6]
    values, mats = _per_system_values(adj, scales, epses, indptr, indices)
    ks = [1, 3, 2, 0, 4]  # ragged, including an empty block
    offs = _ragged_offsets(ks)
    total = int(offs[-1])
    rng = np.random.default_rng(2)
    rhs = np.ascontiguousarray(rng.standard_normal((total, n)))
    xout = np.empty((total, n))
    status = np.empty(len(scales), dtype=np.int64)
    bc.factorize_and_solve_batch(values, rhs, offs, xout, status)
    assert (status == 0).all()
    assert _worst_residual(mats, offs, rhs, xout) < 1e-9


@pytest.mark.skipif(not has_cholmod(), reason="CHOLMOD not built")
def test_cholmod_batch_non_spd_reports_status():
    """A non-SPD system gets a nonzero status; the others still succeed."""
    adj = _grid_adjacency(10)
    m = _sddm(adj, 0.5)
    indptr = m.indptr.astype(np.int64)
    indices = m.indices.astype(np.int64)
    n = m.shape[0]
    bc = cholmod_core().BatchedCholmodSolver(indptr, indices, m.data.astype(np.float64), n)

    base = m.data.astype(np.float64)
    values = np.ascontiguousarray(np.stack([base, base.copy(), base]))
    diag = indices == np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    values[1][diag] = -5.0  # negative diagonal -> not positive definite
    offs = _ragged_offsets([1, 1, 1])
    rng = np.random.default_rng(3)
    rhs = np.ascontiguousarray(rng.standard_normal((3, n)))
    xout = np.empty((3, n))
    status = np.empty(3, dtype=np.int64)
    bc.factorize_and_solve_batch(values, rhs, offs, xout, status)
    assert status[1] != 0
    assert status[0] == 0 and status[2] == 0


@pytest.mark.skipif(not has_cholmod(), reason="CHOLMOD not built")
def test_cholmod_batch_repeated_is_stable():
    """Repeated batched calls (worker reuse) stay correct - thread-safety smoke."""
    adj = _grid_adjacency(11)
    m0 = _sddm(adj, 0.5)
    indptr = m0.indptr.astype(np.int64)
    indices = m0.indices.astype(np.int64)
    n = m0.shape[0]
    bc = cholmod_core().BatchedCholmodSolver(indptr, indices, m0.data.astype(np.float64), n)
    scales = [1.0, 1.5, 0.7, 2.2, 0.9, 1.1]
    values, mats = _per_system_values(adj, scales, [0.5] * len(scales), indptr, indices)
    offs = _ragged_offsets([1] * len(scales))
    rng = np.random.default_rng(4)
    rhs = np.ascontiguousarray(rng.standard_normal((len(scales), n)))
    xout = np.empty((len(scales), n))
    status = np.empty(len(scales), dtype=np.int64)
    for _ in range(25):
        bc.factorize_and_solve_batch(values, rhs, offs, xout, status)
        assert (status == 0).all()
    assert _worst_residual(mats, offs, rhs, xout) < 1e-9
