"""
Tests for the high-level :class:`purc.laplaciansolve.batched.BatchedSDDMSolver`.

Exercises routing (forest vs CHOLMOD, forced routes), the shared-matrix and
per-system batch contracts (against per-system dense solves), their equivalence,
ragged RHS, torch array I/O, and non-SPD error surfacing.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from purc.laplaciansolve._loader import has_cholmod
from purc.laplaciansolve.batched import BatchedSDDMSolver
from purc.laplaciansolve.reference import lap
from purc.laplaciansolve.solver import SolverConfig

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _grid(g):
    """Return a cyclic 4-neighbour grid adjacency on g*g nodes."""
    rows = []
    cols = []
    for i in range(g):
        for j in range(g):
            k = i * g + j
            if i + 1 < g:
                rows += [k, k + g]
                cols += [k + g, k]
            if j + 1 < g:
                rows += [k, k + 1]
                cols += [k + 1, k]
    n = g * g
    return sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _path(n):
    """Return a path adjacency on n nodes (a forest)."""
    r = list(range(n - 1))
    rows = r + [i + 1 for i in r]
    cols = [i + 1 for i in r] + r
    return sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _sddm(adj, eps):
    """Return M = lap(adj) + eps*I as a sorted float64 CSC."""
    m = (lap(adj) + eps * sp.identity(adj.shape[0])).tocsc()
    m.sort_indices()
    return m


def _stack_values(adj, scales, epses, ref):
    """Stack [B, nnz] values of distinct SDDM matrices on ref's pattern."""
    vals = []
    mats = []
    for w, e in zip(scales, epses, strict=True):
        m = _sddm(w * adj, e)
        assert np.array_equal(m.indices, ref.indices) and np.array_equal(m.indptr, ref.indptr)
        vals.append(m.data)
        mats.append(m)
    return np.ascontiguousarray(np.stack(vals)), mats


def test_route_forest_for_tree():
    """A tree (forest) auto-routes to the exact forest backend."""
    s = BatchedSDDMSolver(_sddm(_path(12), 0.3))
    assert s.method == "forest"


def test_route_cholmod_for_cyclic():
    """A cyclic graph auto-routes to CHOLMOD (when available)."""
    if not has_cholmod():
        pytest.skip("CHOLMOD not built")
    s = BatchedSDDMSolver(_sddm(_grid(8), 0.5))
    assert s.method == "cholmod"


def test_force_forest_raises_on_cycle():
    """Forcing the forest route on a cyclic matrix raises."""
    with pytest.raises(ValueError):
        BatchedSDDMSolver(_sddm(_grid(6), 0.5), config=SolverConfig(method="forest"))


@pytest.mark.parametrize("acyclic", [True, False])
def test_shared_matrix_batch_matches_dense(acyclic):
    """Shared-matrix batch (one M, B RHS) matches the dense solve for each RHS."""
    if not acyclic and not has_cholmod():
        pytest.skip("CHOLMOD not built")
    m = _sddm(_path(16), 0.4) if acyclic else _sddm(_grid(8), 0.5)
    s = BatchedSDDMSolver(m)
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal((10, s.n))
    x = s.solve_batch(m.data.astype(np.float64), rhs)
    dense = m.toarray()
    for r in range(10):
        assert np.linalg.norm(dense @ x[r] - rhs[r]) / np.linalg.norm(rhs[r]) < 1e-9


@pytest.mark.parametrize("acyclic", [True, False])
def test_per_system_batch_matches_dense_ragged(acyclic):
    """Per-system batch (B matrices, ragged RHS) matches each system's dense solve."""
    if not acyclic and not has_cholmod():
        pytest.skip("CHOLMOD not built")
    adj = _path(18) if acyclic else _grid(9)
    ref = _sddm(adj, 0.5)
    s = BatchedSDDMSolver(ref)
    scales = [1.0, 2.0, 0.5, 1.4]
    epses = [0.5, 0.2, 0.9, 0.3]
    values, mats = _stack_values(adj, scales, epses, ref)
    offs = np.array([0, 2, 3, 6, 6], dtype=np.int64)  # ragged 2,1,3,0
    total = int(offs[-1])
    rng = np.random.default_rng(1)
    rhs = rng.standard_normal((total, s.n))
    x = s.solve_batch(values, rhs, rhs_offsets=offs)
    for sysid, m in enumerate(mats):
        dense = m.toarray()
        for r in range(int(offs[sysid]), int(offs[sysid + 1])):
            assert np.linalg.norm(dense @ x[r] - rhs[r]) / np.linalg.norm(rhs[r]) < 1e-9


@pytest.mark.parametrize("acyclic", [True, False])
def test_shared_equals_per_system(acyclic):
    """Feeding the same matrix B times per-system equals the shared-matrix path."""
    if not acyclic and not has_cholmod():
        pytest.skip("CHOLMOD not built")
    m = _sddm(_path(14), 0.6) if acyclic else _sddm(_grid(7), 0.5)
    s = BatchedSDDMSolver(m)
    rng = np.random.default_rng(2)
    rhs = rng.standard_normal((5, s.n))
    x_shared = s.solve_batch(m.data.astype(np.float64), rhs)
    values = np.ascontiguousarray(np.tile(m.data.astype(np.float64), (5, 1)))
    x_per = s.solve_batch(values, rhs)  # one RHS per system, offsets default to arange
    assert np.allclose(x_shared, x_per, atol=1e-10)


def test_per_system_one_rhs_each_default_offsets():
    """Per-system with [B, n] rhs and no offsets solves one RHS per system."""
    adj = _path(10)
    ref = _sddm(adj, 0.5)
    s = BatchedSDDMSolver(ref)
    values, mats = _stack_values(adj, [1.0, 2.0, 3.0], [0.5, 0.3, 0.7], ref)
    rng = np.random.default_rng(5)
    rhs = rng.standard_normal((3, s.n))
    x = s.solve_batch(values, rhs)
    for sysid, m in enumerate(mats):
        assert (
            np.linalg.norm(m.toarray() @ x[sysid] - rhs[sysid]) / np.linalg.norm(rhs[sysid]) < 1e-9
        )


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_io_round_trips():
    """A torch RHS returns a torch solution on the same device/dtype."""
    m = _sddm(_path(12), 0.4)
    s = BatchedSDDMSolver(m)
    rhs = torch.randn(6, s.n, dtype=torch.float64)
    x = s.solve_batch(m.data.astype(np.float64), rhs)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (6, s.n) and x.dtype == torch.float64
    resid = (torch.from_numpy(m.toarray()) @ x.T - rhs.T).norm() / rhs.norm()
    assert float(resid) < 1e-9


def test_pattern_mismatch_raises():
    """Passing values with the wrong nnz raises a clear error."""
    m = _sddm(_path(10), 0.5)
    s = BatchedSDDMSolver(m)
    with pytest.raises(ValueError):
        s.solve_batch(np.ones(s.nnz + 2), np.zeros((1, s.n)))


@pytest.mark.skipif(not has_cholmod(), reason="CHOLMOD not built")
def test_non_spd_system_raises():
    """A non-SPD system in a CHOLMOD batch raises LinAlgError."""
    adj = _grid(8)
    ref = _sddm(adj, 0.5)
    s = BatchedSDDMSolver(ref)
    base = ref.data.astype(np.float64)
    values = np.ascontiguousarray(np.stack([base, base.copy()]))
    diag = ref.indices == np.repeat(np.arange(ref.shape[0]), np.diff(ref.indptr))
    values[1][diag] = -3.0  # not positive definite
    rng = np.random.default_rng(7)
    rhs = rng.standard_normal((2, s.n))
    with pytest.raises(np.linalg.LinAlgError):
        s.solve_batch(values, rhs)
