"""
Adversarial stress tests: PURC interface vs an independent naive oracle.

Correctness here is checked against a reference that shares *nothing* with the
native path: it assembles ``M = C diag(D) C^T + eps*I`` from scratch with a plain
Python loop and solves every right-hand side with a dense ``numpy.linalg.solve``
(and, for larger systems, scipy ``spsolve``).  Every case the PURC API supports -
shared / per-system / ragged batches, active masks, scalar / per-system eps, the
forest and CHOLMOD routes - plus the awkward edge cases (single node, single
edge, isolated nodes, disconnected graphs, all-/no-active masks, B=1, empty RHS
blocks, near-singular eps, larger random graphs) is swept over many seeds and
compared element-wise to that oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from purc.laplaciansolve._loader import has_cholmod
from purc.laplaciansolve.batched import BatchedSDDMSolver
from purc.laplaciansolve.purc import PURCLaplacianSolver
from purc.laplaciansolve.solver import SolverConfig

# --------------------------------------------------------------------------- #
# Independent oracle (no native code, no shared helpers with the solver).
# --------------------------------------------------------------------------- #


def naive_matrix(edges, n, w_eff, eps):
    """Assemble M = C diag(w_eff) C^T + eps*I from scratch (plain Python loop)."""
    matrix = np.zeros((n, n), dtype=np.float64)
    for e in range(edges.shape[0]):
        u = int(edges[e, 0])
        v = int(edges[e, 1])
        w = float(w_eff[e])
        matrix[u, u] += w
        matrix[v, v] += w
        matrix[u, v] -= w
        matrix[v, u] -= w
    for i in range(n):
        matrix[i, i] += float(eps)
    return matrix


def naive_solve(edges, n, weights, mask, eps, rhs, offsets):
    """Solve every system/RHS naively (dense or scipy), mirroring the PURC contract."""
    w = np.asarray(weights, dtype=np.float64)
    if mask is not None:
        w = w * np.asarray(mask, dtype=np.float64)
    shared = w.ndim == 1
    batch = 1 if shared else w.shape[0]
    eps_arr = np.ravel(np.asarray(eps, dtype=np.float64))
    eps_v = np.full(batch, eps_arr[0]) if eps_arr.size == 1 else eps_arr
    rhs = np.atleast_2d(np.asarray(rhs, dtype=np.float64))
    total = rhs.shape[0]
    out = np.empty((total, n), dtype=np.float64)

    def solve_block(matrix, rows):
        if n <= 48:
            for r in rows:
                out[r] = np.linalg.solve(matrix, rhs[r])
        else:
            lu = spla.splu(sp.csc_matrix(matrix))
            for r in rows:
                out[r] = lu.solve(rhs[r])

    if shared:
        solve_block(naive_matrix(edges, n, w, eps_v[0]), range(total))
    else:
        offs = np.arange(batch + 1) if offsets is None else np.asarray(offsets)
        for s in range(batch):
            solve_block(naive_matrix(edges, n, w[s], eps_v[s]), range(int(offs[s]), int(offs[s + 1])))
    return out


def assert_matches_oracle(edges, n, weights, *, mask=None, eps, rhs, offsets=None, config=None):
    """Solve via PURCLaplacianSolver and assert agreement with the naive oracle."""
    solver = PURCLaplacianSolver(edges, n, config=config)
    res = solver.solve_batch(weights, active_mask=mask, eps=eps, rhs=rhs, rhs_offsets=offsets)
    got = np.atleast_2d(np.asarray(res["solution"], dtype=np.float64))
    ref = naive_solve(edges, n, weights, mask, eps, rhs, offsets)
    # Conditioning-aware tolerance: ill-conditioned M (tiny eps) limits agreement.
    min_eps = float(np.min(np.ravel(np.asarray(eps, dtype=np.float64))))
    atol = max(1e-9, 1e-12 / max(min_eps, 1e-12))
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=atol)
    return solver, got, ref


# --------------------------------------------------------------------------- #
# Graph generators.
# --------------------------------------------------------------------------- #


def random_graph(n, n_extra, rng):
    """A connected random simple graph: a spanning tree plus n_extra extra edges."""
    edges = set()
    order = rng.permutation(n)
    for i in range(1, n):
        j = int(order[rng.integers(0, i)])
        a, b = int(order[i]), j
        edges.add((min(a, b), max(a, b)))
    tries = 0
    while len(edges) < (n - 1) + n_extra and tries < 50 * (n_extra + 1):
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if a != b:
            edges.add((min(a, b), max(a, b)))
        tries += 1
    return np.array(sorted(edges), dtype=np.int64)


def disjoint_union(g1, n1, g2, n2):
    """Concatenate two graphs onto disjoint node sets (a disconnected graph)."""
    return np.vstack([g1, g2 + n1]), n1 + n2


# --------------------------------------------------------------------------- #
# Edge cases.
# --------------------------------------------------------------------------- #


def test_single_node_no_edges():
    """n=1, m=0: M = [eps]; the solve is b/eps."""
    edges = np.empty((0, 2), dtype=np.int64)
    rhs = np.array([[2.0], [5.0]])  # B=2 RHS (shared scalar matrix)
    solver, got, ref = assert_matches_oracle(edges, 1, np.empty(0), eps=0.25, rhs=rhs)
    assert np.allclose(got.ravel(), rhs.ravel() / 0.25)


def test_single_edge():
    """n=2, m=1: the smallest nontrivial forest."""
    edges = np.array([[0, 1]], dtype=np.int64)
    rng = np.random.default_rng(0)
    assert_matches_oracle(edges, 2, np.array([1.7]), eps=0.3, rhs=rng.standard_normal((4, 2)))


def test_star_graph():
    """A star (hub + leaves) is a forest."""
    n = 8
    edges = np.array([[0, i] for i in range(1, n)], dtype=np.int64)
    rng = np.random.default_rng(1)
    s, _, _ = assert_matches_oracle(
        edges, n, rng.uniform(0.5, 2.0, n - 1), eps=0.1, rhs=rng.standard_normal((5, n))
    )
    assert s.phase == "forest"


def test_isolated_node_present():
    """A node with no incident edges (diagonal = eps) is handled."""
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)  # node 4 isolated
    n = 5
    rng = np.random.default_rng(2)
    assert_matches_oracle(edges, n, rng.uniform(0.5, 2.0, 3), eps=0.2, rhs=rng.standard_normal((3, n)))


def test_disconnected_forest():
    """Two disjoint trees form a (disconnected) forest."""
    rng = np.random.default_rng(3)
    g1 = random_graph(5, 0, rng)
    g2 = random_graph(6, 0, rng)
    edges, n = disjoint_union(g1, 5, g2, 6)
    s, _, _ = assert_matches_oracle(
        edges, n, rng.uniform(0.5, 2.0, edges.shape[0]), eps=0.3, rhs=rng.standard_normal((4, n))
    )
    assert s.phase == "forest"


def test_disconnected_cyclic():
    """Two disjoint graphs with cycles route to CHOLMOD."""
    if not has_cholmod():
        pytest.skip("CHOLMOD not built")
    rng = np.random.default_rng(4)
    g1 = random_graph(6, 3, rng)
    g2 = random_graph(7, 4, rng)
    edges, n = disjoint_union(g1, 6, g2, 7)
    assert_matches_oracle(
        edges, n, rng.uniform(0.5, 2.0, edges.shape[0]), eps=0.4, rhs=rng.standard_normal((5, n))
    )


def test_mask_all_inactive_gives_eps_identity():
    """A fully inactive mask makes M = eps*I; the solve is b/eps."""
    rng = np.random.default_rng(5)
    edges = random_graph(10, 5, rng)
    m = edges.shape[0]
    rhs = rng.standard_normal((3, 10))
    assert_matches_oracle(
        edges, 10, rng.uniform(0.5, 2.0, m), mask=np.zeros(m), eps=0.5, rhs=rhs
    )


def test_mask_all_active_equals_no_mask():
    """An all-ones mask equals passing no mask."""
    rng = np.random.default_rng(6)
    edges = random_graph(12, 6, rng)
    m = edges.shape[0]
    w = rng.uniform(0.5, 2.0, m)
    rhs = rng.standard_normal((4, 12))
    s = PURCLaplacianSolver(edges, 12)
    a = np.asarray(s.solve_batch(w, eps=0.3, rhs=rhs)["solution"])
    b = np.asarray(s.solve_batch(w, active_mask=np.ones(m), eps=0.3, rhs=rhs)["solution"])
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_batch_of_one():
    """B=1 per-system batch behaves like a single solve."""
    rng = np.random.default_rng(7)
    edges = random_graph(9, 4, rng)
    m = edges.shape[0]
    assert_matches_oracle(
        edges, 9, rng.uniform(0.5, 2.0, (1, m)), eps=np.array([0.3]), rhs=rng.standard_normal((1, 9))
    )


def test_ragged_with_empty_block():
    """A ragged per-system batch including a K_s=0 block is handled."""
    rng = np.random.default_rng(8)
    edges = random_graph(11, 5, rng)
    m = edges.shape[0]
    weights = rng.uniform(0.5, 2.0, (4, m))
    offs = np.array([0, 2, 2, 5, 7], dtype=np.int64)  # K = 2, 0, 3, 2
    rhs = rng.standard_normal((7, 11))
    assert_matches_oracle(
        edges, 11, weights, eps=np.array([0.3, 0.2, 0.5, 0.4]), rhs=rhs, offsets=offs
    )


def test_near_singular_eps_residual():
    """Tiny eps is ill-conditioned: check the residual against the assembled M."""
    rng = np.random.default_rng(9)
    edges = random_graph(20, 10, rng)
    m = edges.shape[0]
    w = rng.uniform(0.5, 2.0, m)
    eps = 1e-9
    rhs = rng.standard_normal((4, 20))
    s = PURCLaplacianSolver(edges, 20)
    x = np.asarray(s.solve_batch(w, eps=eps, rhs=rhs)["solution"])
    matrix = naive_matrix(edges, 20, w, eps)
    for r in range(4):
        assert np.linalg.norm(matrix @ x[r] - rhs[r]) / np.linalg.norm(rhs[r]) < 1e-6


# --------------------------------------------------------------------------- #
# Randomized sweep over topologies x batch modes x masks x eps.
# --------------------------------------------------------------------------- #

_SWEEP = [
    (seed, n, extra, mode)
    for seed in range(6)
    for (n, extra) in [(1, 0), (2, 0), (5, 0), (12, 0), (12, 8), (30, 20), (50, 40)]
    for mode in ("shared", "per_system", "ragged")
]


@pytest.mark.parametrize("seed,n,extra,mode", _SWEEP)
def test_random_sweep_matches_oracle(seed, n, extra, mode):
    """Sweep random graphs/weights/masks/eps and match the naive oracle exactly."""
    if extra > 0 and not has_cholmod():
        pytest.skip("CHOLMOD not built (cyclic graph)")
    rng = np.random.default_rng(1000 + seed)
    edges = random_graph(n, extra, rng) if n > 1 else np.empty((0, 2), dtype=np.int64)
    m = edges.shape[0]
    use_mask = bool(rng.integers(0, 2)) and m > 0

    if mode == "shared":
        weights = rng.uniform(0.3, 2.5, m)
        eps = float(rng.uniform(1e-3, 1.0))
        rhs = rng.standard_normal((int(rng.integers(1, 5)), n))
        offsets = None
        mask = (rng.random(m) > 0.3).astype(float) if use_mask else None
    else:
        batch = int(rng.integers(1, 5))
        weights = rng.uniform(0.3, 2.5, (batch, m))
        eps = rng.uniform(1e-3, 1.0, batch)
        mask = (rng.random((batch, m)) > 0.3).astype(float) if use_mask else None
        if mode == "per_system":
            rhs = rng.standard_normal((batch, n))
            offsets = None
        else:  # ragged
            ks = rng.integers(0, 4, batch)
            if ks.sum() == 0:
                ks[0] = 1
            offsets = np.concatenate([[0], np.cumsum(ks)]).astype(np.int64)
            rhs = rng.standard_normal((int(offsets[-1]), n))

    assert_matches_oracle(edges, n, weights, mask=mask, eps=eps, rhs=rhs, offsets=offsets)


# --------------------------------------------------------------------------- #
# BatchedSDDMSolver directly vs naive (isolates the solve from PURC assembly).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(4))
def test_batched_solver_values_match_oracle(seed):
    """Feed BatchedSDDMSolver raw per-system CSC values; match the naive solve."""
    rng = np.random.default_rng(2000 + seed)
    n, extra = 15, 9
    if not has_cholmod():
        pytest.skip("CHOLMOD not built")
    edges = random_graph(n, extra, rng)
    m = edges.shape[0]
    # Build matrices and their CSC values in a fixed shared pattern.
    base = naive_matrix(edges, n, np.ones(m), 1.0)
    pattern = sp.csc_matrix(base)
    pattern.sort_indices()
    solver = BatchedSDDMSolver(pattern)
    batch = 3
    mats = [sp.csc_matrix(naive_matrix(edges, n, rng.uniform(0.5, 2.0, m), float(rng.uniform(0.1, 1.0)))) for _ in range(batch)]
    for mat in mats:
        mat.sort_indices()
        assert np.array_equal(mat.indices, pattern.indices) and np.array_equal(mat.indptr, pattern.indptr)
    values = np.ascontiguousarray(np.stack([mat.data for mat in mats]))
    rhs = rng.standard_normal((batch, n))
    got = np.asarray(solver.solve_batch(values, rhs))  # BatchedSDDMSolver returns the array
    for s in range(batch):
        ref = np.linalg.solve(mats[s].toarray(), rhs[s])
        np.testing.assert_allclose(got[s], ref, rtol=1e-7, atol=1e-9)


# --------------------------------------------------------------------------- #
# Reuse, scale, and dtype robustness.
# --------------------------------------------------------------------------- #


def test_repeated_drifting_weights_reuse():
    """Many sequential solves on one handle (drifting weights) each match naive."""
    rng = np.random.default_rng(3000)
    edges = random_graph(25, 15 if has_cholmod() else 0, rng)
    m = edges.shape[0]
    solver = PURCLaplacianSolver(edges, 25)
    w = rng.uniform(0.5, 2.0, m)
    for _ in range(20):  # SSN-like sequence: handle + factor workspaces reused
        w = np.clip(w + 0.1 * rng.standard_normal(m), 0.1, 3.0)
        eps = float(rng.uniform(1e-2, 1.0))
        rhs = rng.standard_normal((3, 25))
        got = np.asarray(solver.solve_batch(w, eps=eps, rhs=rhs)["solution"])
        ref = naive_solve(edges, 25, w, None, eps, rhs, None)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_large_per_system_batch():
    """A large per-system batch (B=64 distinct matrices) matches naive."""
    if not has_cholmod():
        pytest.skip("CHOLMOD not built")
    rng = np.random.default_rng(3001)
    n = 30
    edges = random_graph(n, 18, rng)
    m = edges.shape[0]
    batch = 64
    weights = rng.uniform(0.5, 2.0, (batch, m))
    eps = rng.uniform(1e-2, 1.0, batch)
    rhs = rng.standard_normal((batch, n))
    assert_matches_oracle(edges, n, weights, eps=eps, rhs=rhs)


def test_float32_and_noncontiguous_inputs():
    """float32 / non-contiguous (Fortran-order) weights and RHS still solve correctly."""
    rng = np.random.default_rng(3002)
    edges = random_graph(14, 7 if has_cholmod() else 0, rng)
    m = edges.shape[0]
    solver = PURCLaplacianSolver(edges, 14)
    w32 = rng.uniform(0.5, 2.0, m).astype(np.float32)
    rhs = np.asfortranarray(rng.standard_normal((5, 14)))  # non-contiguous
    got = np.asarray(solver.solve_batch(w32, eps=0.3, rhs=rhs)["solution"])
    ref = naive_solve(edges, 14, w32.astype(np.float64), None, 0.3, np.ascontiguousarray(rhs), None)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_torch_input_matches_oracle():
    """A torch RHS/weights solve matches the numpy naive oracle."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3003)
    edges = random_graph(16, 9 if has_cholmod() else 0, rng)
    m = edges.shape[0]
    solver = PURCLaplacianSolver(edges, 16)
    w = rng.uniform(0.5, 2.0, m)
    rhs = rng.standard_normal((4, 16))
    got = solver.solve_batch(torch.from_numpy(w), eps=0.4, rhs=torch.from_numpy(rhs))["solution"]
    assert isinstance(got, torch.Tensor)
    ref = naive_solve(edges, 16, w, None, 0.4, rhs, None)
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-6, atol=1e-8)
