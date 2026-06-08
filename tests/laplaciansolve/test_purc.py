"""
Tests for :class:`purc.laplaciansolve.purc.PURCLaplacianSolver`.

Validates the native incidence assembly against a scipy oracle, the solve-step /
batch residuals, active-mask deactivation, per-system (distinct-matrix) batches,
routing/phase, edge-list validation, torch I/O, and an end-to-end synthetic
semismooth-Newton sequence.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from purc.laplaciansolve.purc import PURCLaplacianSolver
from purc.laplaciansolve.reference import lap
from purc.laplaciansolve.solver import SolverConfig

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _grid_edges(g):
    """Return (n, edges) for a g x g 4-neighbour grid (cyclic)."""
    edges = []
    for i in range(g):
        for j in range(g):
            k = i * g + j
            if i + 1 < g:
                edges.append((k, k + g))
            if j + 1 < g:
                edges.append((k, k + 1))
    return g * g, np.array(edges, dtype=np.int64)


def _path_edges(n):
    """Return edges of a path on n nodes (a forest)."""
    return np.array([(i, i + 1) for i in range(n - 1)], dtype=np.int64)


def _ref_matrix(edges, n, weights, eps):
    """Assemble M = C diag(weights) C^T + eps*I with scipy (oracle)."""
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adj = sp.csc_matrix((np.concatenate([weights, weights]), (rows, cols)), shape=(n, n))
    return (lap(adj) + eps * sp.identity(n)).tocsc()


def test_native_assembly_matches_scipy_oracle():
    """The native assembler reproduces scipy's lap(adj)+eps*I in CSC order."""
    n, edges = _grid_edges(7)
    m = edges.shape[0]
    s = PURCLaplacianSolver(edges, n)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, m)
    eps = 0.123
    values = s._assemble(w[None, :], np.array([eps]))[0]
    ref = _ref_matrix(edges, n, w, eps)
    # s._solver's pattern is the cached one; compare values in that order.
    assert values.shape[0] == s._nnz
    assembled = sp.csc_matrix((values, s._indices, s._indptr), shape=(n, n))
    assert np.allclose(assembled.toarray(), ref.toarray(), atol=1e-12)


def test_solve_step_residual():
    """solve_step returns a length-n step solving the assembled system."""
    n, edges = _grid_edges(8)
    m = edges.shape[0]
    s = PURCLaplacianSolver(edges, n)
    rng = np.random.default_rng(1)
    w = rng.uniform(0.5, 2.0, m)
    b = rng.standard_normal(n)
    res = s.solve_step(w, eps=1e-3, rhs=b)
    assert set(res) >= {"solution", "method", "phase", "converged", "iterations"}
    x = res["solution"]
    assert x.shape == (n,)
    ref = _ref_matrix(edges, n, w, 1e-3).toarray()
    assert np.linalg.norm(ref @ x - b) / np.linalg.norm(b) < 1e-9


def test_active_mask_deactivates_edges():
    """An active mask zeroes inactive edges (matches a mask-folded reference)."""
    n, edges = _grid_edges(7)
    m = edges.shape[0]
    s = PURCLaplacianSolver(edges, n)
    rng = np.random.default_rng(2)
    w = rng.uniform(0.5, 2.0, m)
    mask = (rng.random(m) > 0.3).astype(float)
    eps = 1e-2
    b = rng.standard_normal(n)
    x = s.solve_step(w, active_mask=mask, eps=eps, rhs=b)["solution"]
    ref = _ref_matrix(edges, n, w * mask, eps).toarray()
    assert np.linalg.norm(ref @ x - b) / np.linalg.norm(b) < 1e-9


def test_per_system_batch_matches_dense_ragged():
    """Per-system batch (distinct weights/eps, ragged RHS) solves each system."""
    n, edges = _grid_edges(9)
    m = edges.shape[0]
    s = PURCLaplacianSolver(edges, n)
    rng = np.random.default_rng(3)
    weights = rng.uniform(0.5, 2.0, (4, m))
    epses = np.array([1e-2, 1e-3, 1e-1, 1e-2])
    offs = np.array([0, 2, 3, 6, 6], dtype=np.int64)  # ragged 2,1,3,0
    total = int(offs[-1])
    rhs = rng.standard_normal((total, n))
    res = s.solve_batch(weights, eps=epses, rhs=rhs, rhs_offsets=offs)
    x = res["solution"]
    for sysid in range(4):
        ref = _ref_matrix(edges, n, weights[sysid], epses[sysid]).toarray()
        for r in range(int(offs[sysid]), int(offs[sysid + 1])):
            assert np.linalg.norm(ref @ x[r] - rhs[r]) / np.linalg.norm(rhs[r]) < 1e-9


def test_shared_batch_matches_dense():
    """Shared-matrix batch (one weight set, B RHS) solves every RHS."""
    n, edges = _grid_edges(8)
    m = edges.shape[0]
    s = PURCLaplacianSolver(edges, n)
    rng = np.random.default_rng(4)
    w = rng.uniform(0.5, 2.0, m)
    rhs = rng.standard_normal((10, n))
    x = s.solve_batch(w, eps=1e-2, rhs=rhs)["solution"]
    ref = _ref_matrix(edges, n, w, 1e-2).toarray()
    for r in range(10):
        assert np.linalg.norm(ref @ x[r] - rhs[r]) / np.linalg.norm(rhs[r]) < 1e-9


def test_routing_phase():
    """A tree routes to forest; a cyclic grid routes to cyclic (CHOLMOD)."""
    s_tree = PURCLaplacianSolver(_path_edges(20), 20)
    assert s_tree.method == "forest" and s_tree.phase == "forest"
    n, edges = _grid_edges(6)
    s_grid = PURCLaplacianSolver(edges, n)
    assert s_grid.phase in ("cyclic", "forest")  # cyclic if CHOLMOD built


def test_force_forest_on_cyclic_raises():
    """Forcing the forest route on a cyclic network raises."""
    n, edges = _grid_edges(5)
    with pytest.raises(ValueError):
        PURCLaplacianSolver(edges, n, config=SolverConfig(method="forest"))


@pytest.mark.parametrize(
    "bad_edges",
    [
        np.array([[0, 0]], dtype=np.int64),  # self-loop
        np.array([[0, 1], [1, 0]], dtype=np.int64),  # duplicate undirected
        np.array([[0, 9]], dtype=np.int64),  # out of range
    ],
)
def test_edge_validation(bad_edges):
    """Malformed edge lists are rejected with a clear error."""
    with pytest.raises(ValueError):
        PURCLaplacianSolver(bad_edges, 4)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_io_round_trips():
    """Torch weights and RHS return a torch solution."""
    n, edges = _grid_edges(7)
    m = edges.shape[0]
    s = PURCLaplacianSolver(edges, n)
    w = torch.rand(m, dtype=torch.float64) + 0.5
    b = torch.randn(n, dtype=torch.float64)
    res = s.solve_step(w, eps=1e-2, rhs=b)
    x = res["solution"]
    assert isinstance(x, torch.Tensor) and x.shape == (n,)
    ref = _ref_matrix(edges, n, w.numpy(), 1e-2).toarray()
    assert np.linalg.norm(ref @ x.numpy() - b.numpy()) / np.linalg.norm(b.numpy()) < 1e-9


def test_synthetic_newton_sequence_converges():
    """A drifting-weights SSN sequence solves each step to the conditioning floor."""
    n, edges = _grid_edges(8)
    m = edges.shape[0]
    lengths = np.linspace(1.0, 3.0, m)
    s = PURCLaplacianSolver(edges, n)
    rng = np.random.default_rng(5)
    x_hat = rng.uniform(0.2, 0.8, m)
    for _ in range(6):
        x_hat = np.clip(x_hat + 0.05 * rng.standard_normal(m), 0.05, 0.95)
        weights = (1.0 + x_hat) / lengths
        r = rng.standard_normal(n)
        eps_k = max(min(1e-6, float(np.linalg.norm(r))), 1e-8)
        x = s.solve_step(weights, eps=eps_k, rhs=-r)["solution"]
        ref = _ref_matrix(edges, n, weights, eps_k).toarray()
        assert np.linalg.norm(ref @ x + r) / np.linalg.norm(r) < 1e-6


def _incidence(edges, n):
    """Build a node-link incidence C (n x m): column e has +1 at u_e, -1 at v_e."""
    m = edges.shape[0]
    data = np.concatenate([np.ones(m), -np.ones(m)])
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.tile(np.arange(m), 2)
    return sp.csc_matrix((data, (rows, cols)), shape=(n, m))


def _adjacency(edges, n):
    """Build a symmetric 0/1 adjacency from an edge list."""
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    return sp.csc_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))


def test_accepts_edge_list_adjacency_incidence():
    """Edge list, adjacency, and incidence inputs yield the same M (uniform weights)."""
    n, edges = _grid_edges(6)
    m = edges.shape[0]
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal((4, n))
    builders = [
        PURCLaplacianSolver(edges, n),
        PURCLaplacianSolver(_adjacency(edges, n)),  # square -> adjacency
        PURCLaplacianSolver(_incidence(edges, n)),  # rectangular -> incidence
        PURCLaplacianSolver.from_adjacency(_adjacency(edges, n)),
        PURCLaplacianSolver.from_incidence(_incidence(edges, n)),
    ]
    sols = []
    for s in builders:
        assert s.n == n and s.num_edges == m
        # uniform weights => identical M across forms regardless of edge ordering
        sols.append(
            np.asarray(s.solve_batch(np.full(s.num_edges, 1.3), eps=0.2, rhs=rhs)["solution"])
        )
    for other in sols[1:]:
        assert np.allclose(sols[0], other, atol=1e-10)


def test_dense_adjacency_accepted():
    """A dense numpy adjacency is accepted (square)."""
    n, edges = _grid_edges(4)
    s = PURCLaplacianSolver(_adjacency(edges, n).toarray())
    assert s.n == n and s.num_edges == edges.shape[0]


def test_n_nodes_mismatch_with_matrix_raises():
    """A matrix input whose shape contradicts n_nodes is rejected."""
    n, edges = _grid_edges(4)
    with pytest.raises(ValueError):
        PURCLaplacianSolver(_adjacency(edges, n), n + 3)


def test_square_incidence_via_classmethod():
    """A cycle's incidence is square (m == n); from_incidence parses it correctly."""
    n = 6
    edges = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int64)  # cycle: n edges
    inc = _incidence(edges, n)  # n x n (square)
    s = PURCLaplacianSolver.from_incidence(inc)
    assert s.n == n and s.num_edges == n
    rng = np.random.default_rng(1)
    b = rng.standard_normal(n)
    x = s.solve_step(np.ones(n), eps=0.3, rhs=b)["solution"]
    ref = _ref_matrix(edges, n, np.ones(n), 0.3).toarray()
    assert np.linalg.norm(ref @ x - b) / np.linalg.norm(b) < 1e-9
