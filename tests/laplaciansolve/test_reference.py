"""
Stage 0 correctness tests for the pure-Python reference solver.

Validates the approxChol build + PCG (Laplacian path), the SDDM embedding
against an exact dense solve, the exact forest factorization, reproducibility,
and the documented edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    disconnected_graph,
    grid_graph,
    mean_zero,
    random_connected_graph,
    random_tree,
    sddm_from_adjacency,
    tolerance,
)
from scipy import sparse

from purc.laplaciansolve.reference import (
    ApproxCholParams,
    ArrayStream,
    GeneratorStream,
    adj_val_and_excess,
    approx_chol,
    approxchol_lap,
    approxchol_lap_pcg,
    approxchol_sddm,
    flip_index,
    forest_solve,
    is_forest,
    lap,
    ldl_solve,
)

# --------------------------------------------------------------------------- #
# Laplacian PCG: residual oracle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n,p", [(50, 0.1), (200, 0.03), (500, 0.01)])
def test_lap_pcg_residual(n, p):
    """approxChol-preconditioned PCG reaches the requested tolerance."""
    a = random_connected_graph(n, p, seed=n)
    b = mean_zero(np.random.default_rng(n + 1), n)
    result = approxchol_lap_pcg(a, b, tol=1e-8, maxits=1000, seed=7)
    la = lap(a)
    relres = np.linalg.norm(la @ result.x - b) / np.linalg.norm(b)
    assert result.converged
    assert relres < 1e-6
    assert result.iterations < n  # nearly-linear preconditioner => few iters


def test_lap_closure_matches_residual(rng):
    """The high-level closure solves lap(a) x = b for mean-zero b."""
    a = random_connected_graph(150, 0.04, seed=2)
    solve = approxchol_lap(a, tol=1e-8, seed=11)
    b = mean_zero(rng, 150)
    x = solve(b)
    la = lap(a)
    assert np.linalg.norm(la @ x - b) / np.linalg.norm(b) < 1e-6


def test_preconditioner_quality_few_iters():
    """A single approxChol apply already captures most of the inverse."""
    a = random_connected_graph(400, 0.02, seed=4)
    # tol large enough that convergence is governed by preconditioner quality.
    result = approxchol_lap_pcg(a, mean_zero(np.random.default_rng(5), 400), tol=1e-1, seed=4)
    assert result.relres < 1e-1
    assert result.iterations <= 5


# --------------------------------------------------------------------------- #
# SDDM path (the PURC system H + eps I)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-6])
def test_sddm_vs_dense(eps, rng):
    """approxchol_sddm matches an exact dense solve of M = lap(a) + eps I."""
    a = random_connected_graph(120, 0.05, seed=3)
    m = sddm_from_adjacency(a, eps)
    b = rng.standard_normal(120)  # general RHS (no mean-zero requirement)
    x = approxchol_sddm(m, tol=1e-10, maxits=3000, seed=9)(b)
    x_exact = np.linalg.solve(m.toarray(), b)
    atol, rtol = tolerance(np.float64)
    assert np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact) < rtol
    assert np.linalg.norm(m @ x - b) / np.linalg.norm(b) < 1e-7


def test_sddm_purc_excess_equals_eps():
    """For M = lap(a) + eps I every row's dominance excess equals eps."""
    a = random_connected_graph(40, 0.1, seed=1)
    eps = 1e-4
    m = sddm_from_adjacency(a, eps)
    _adj, _diag, excess = adj_val_and_excess(m)
    assert np.allclose(excess, eps, atol=1e-12)


def test_sddm_already_laplacian(rng):
    """An SDDM input with zero excess is treated as a plain Laplacian."""
    a = random_connected_graph(60, 0.08, seed=6)
    la = lap(a)  # exact Laplacian: excess == 0
    b = mean_zero(rng, 60)
    x = approxchol_sddm(sparse.csc_matrix(la), tol=1e-9, seed=2)(b)
    assert np.linalg.norm(la @ x - b) / np.linalg.norm(b) < 1e-6


# --------------------------------------------------------------------------- #
# Forest fast path (exact, deterministic)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [2, 10, 64, 256])
def test_forest_exact(n):
    """The forest LDL solves a tree Laplacian to machine precision."""
    tree = random_tree(n, seed=n)
    assert is_forest(tree)
    b = mean_zero(np.random.default_rng(n), n)
    x = forest_solve(tree, b)
    lt = lap(tree)
    assert np.linalg.norm(lt @ x - b) / np.linalg.norm(b) < 1e-10


def test_approxchol_on_forest_is_exact_and_uses_no_randomness():
    """On a forest, the :deg build eliminates only leaves: exact, RNG-free."""
    tree = random_tree(128, seed=42)
    stream = GeneratorStream(0)
    ldli = approx_chol(tree, ApproxCholParams(order="deg"), rng=stream)
    assert stream.count == 0  # leaves never trigger survivor-edge sampling

    b = mean_zero(np.random.default_rng(1), 128)
    x_approx = ldl_solve(ldli, b)
    x_forest = forest_solve(tree, b)
    assert np.allclose(x_approx, x_forest, atol=1e-9)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_single_edge():
    """The 2-node single-edge Laplacian solves analytically."""
    w = 2.5
    a = sparse.csc_matrix(np.array([[0.0, w], [w, 0.0]]))
    b = np.array([1.0, -1.0])
    x = approxchol_lap(a, tol=1e-12, seed=0)(b)
    # lap = [[w,-w],[-w,w]]; mean-zero solution is [0.5/w, -0.5/w].
    assert np.allclose(x, [0.5 / w, -0.5 / w], atol=1e-9)


def test_disconnected_graph(rng):
    """Disconnected graphs solve per component; singletons return zeros."""
    a, n_comp = disconnected_graph(seed=0)
    n = a.shape[0]
    solve = approxchol_lap(a, tol=1e-10, seed=3)
    # Per-component mean-zero RHS so each singular block is consistent.
    b = np.zeros(n)
    b[0:3] = mean_zero(rng, 3)
    b[3:6] = mean_zero(rng, 3)
    b[6] = 0.0  # isolated vertex
    x = solve(b)
    la = lap(a)
    # Residual is checked per connected block (the global operator is singular).
    assert np.linalg.norm((la @ x - b)[0:3]) < 1e-6
    assert np.linalg.norm((la @ x - b)[3:6]) < 1e-6
    assert x[6] == 0.0  # nullSolver leaves the isolated vertex at zero


def test_explicit_zero_weight_entries_are_ignored(rng):
    """Stored structural zeros in the adjacency do not corrupt the factorization."""
    a = random_connected_graph(40, 0.1, seed=8).tolil()
    a[0, 5] = 0.0  # explicit zero entry (kept in the structure)
    a[5, 0] = 0.0
    a = sparse.csc_matrix(a)
    b = mean_zero(rng, 40)
    x = approxchol_lap(a, tol=1e-9, seed=8)(b)
    la = lap(a)
    assert np.linalg.norm(la @ x - b) / np.linalg.norm(b) < 1e-6


def test_near_singular_regularization(rng):
    """A tiny eps still yields a solvable SDDM system via grounding."""
    a = random_connected_graph(80, 0.05, seed=7)
    m = sddm_from_adjacency(a, 1e-8)
    b = rng.standard_normal(80)
    x = approxchol_sddm(m, tol=1e-10, maxits=5000, seed=7)(b)
    assert np.linalg.norm(m @ x - b) / np.linalg.norm(b) < 1e-6


def test_grid_graph_converges(rng):
    """A cyclic 2-D grid (nontrivial fill-in) still converges quickly."""
    a = grid_graph(20, 20, seed=1)
    n = a.shape[0]
    result = approxchol_lap_pcg(a, mean_zero(rng, n), tol=1e-8, seed=1)
    assert result.converged
    assert np.isfinite(result.relres)


# --------------------------------------------------------------------------- #
# Structural / reproducibility properties
# --------------------------------------------------------------------------- #


def test_flip_index_symmetry():
    """flip_index pairs each nonzero with its transpose of equal weight."""
    a = random_connected_graph(30, 0.15, seed=5)
    flip = flip_index(a)
    rows = a.indices
    cols = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    for p in range(a.nnz):
        q = flip[p]
        assert rows[q] == cols[p] and cols[q] == rows[p]
        assert np.isclose(a.data[p], a.data[q])
        assert flip[q] == p  # involution


def test_reproducible_with_seed():
    """Two builds with the same seed produce an identical factorization."""
    a = random_connected_graph(100, 0.05, seed=9)
    l1 = approx_chol(a, rng=123)
    l2 = approx_chol(a, rng=123)
    assert np.array_equal(l1.col, l2.col)
    assert np.array_equal(l1.colptr, l2.colptr)
    assert np.array_equal(l1.rowval, l2.rowval)
    assert np.allclose(l1.fval, l2.fval)
    assert np.allclose(l1.d, l2.d)


def test_array_stream_determinism():
    """A replayed sample stream yields a deterministic factorization."""
    a = random_connected_graph(60, 0.08, seed=2)
    samples = np.random.default_rng(0).random(10_000)
    l1 = approx_chol(a, rng=ArrayStream(samples))
    l2 = approx_chol(a, rng=ArrayStream(samples))
    assert np.array_equal(l1.rowval, l2.rowval)
    assert np.allclose(l1.fval, l2.fval)
    assert np.allclose(l1.d, l2.d)


def test_pq_pops_min_degree_first():
    """The priority queue returns a minimum-degree vertex first."""
    from purc.laplaciansolve.reference import ApproxCholPQ, build_llmatp

    a = random_connected_graph(50, 0.1, seed=4)
    llmat = build_llmatp(a)
    pq = ApproxCholPQ(llmat.degs)
    first = pq.pop()
    assert llmat.degs[first] == int(llmat.degs.min())


# --------------------------------------------------------------------------- #
# Parameter validation (no silent fallback)
# --------------------------------------------------------------------------- #


def test_params_validation():
    """ApproxCholParams rejects unknown orderings and negative counts."""
    with pytest.raises(ValueError):
        ApproxCholParams(order="bogus")
    with pytest.raises(ValueError):
        ApproxCholParams(stag_test=-1)


@pytest.mark.parametrize("order", ["given", "wdeg"])
def test_unimplemented_orders_raise(order):
    """Not-yet-ported orderings raise clearly rather than silently fall back."""
    a = random_connected_graph(20, 0.2, seed=1)
    with pytest.raises(NotImplementedError):
        approx_chol(a, ApproxCholParams(order=order))


def test_split_merge_unimplemented_raises():
    """split/merge > 0 is rejected until ported (no silent fallback)."""
    a = random_connected_graph(20, 0.2, seed=1)
    with pytest.raises(NotImplementedError):
        approx_chol(a, ApproxCholParams(order="deg", split=2, merge=2))
