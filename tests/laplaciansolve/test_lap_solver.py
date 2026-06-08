"""
Tests for the reusable, auto-caching, batched solver handle.

Mimics the PURC use cases the handle is built for: build once and reuse the
preconditioner across many solves, solve a batch of right-hand sides in
parallel, re-solve after a weight update *without rebuilding*, warm-start, and
the exact forest fast path.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import grid_graph, mean_zero, random_connected_graph, random_tree

pytest.importorskip(
    "purc.laplaciansolve._laplaciansolve_core",
    reason="native core not built (run: pip install --no-build-isolation -e .)",
)

from purc.laplaciansolve import LaplacianSolver, SolverConfig  # noqa: E402
from purc.laplaciansolve.reference import lap  # noqa: E402


def _relres(a, x, b):
    """
    Relative Laplacian residual ``||lap(a) x - (b - mean b)|| / ||b - mean b||``.

    Args:
        a: Adjacency matrix.
        x: Candidate solution.
        b: Right-hand side.

    Returns:
        The relative residual.

    """
    la = lap(a)
    bz = b - b.mean()
    return np.linalg.norm(la @ x - bz) / np.linalg.norm(bz)


def _reweight(a, seed):
    """
    Return a copy of *a* with new positive weights on the SAME pattern.

    Perturbs symmetrically via per-node factors (``w_ij -> w_ij * f_i * f_j``),
    which keeps ``a[i, j] == a[j, i]`` (an asymmetric adjacency would not be a
    valid Laplacian and PCG would not converge).

    Args:
        a: Base adjacency (CSC).
        seed: RNG seed.

    Returns:
        A reweighted, still-symmetric adjacency sharing ``a``'s pattern.

    """
    r = np.random.default_rng(seed)
    n = a.shape[0]
    f = np.abs(1.0 + 0.1 * r.standard_normal(n)) + 0.05
    out = a.copy()
    rows = out.indices  # row of each CSC data entry
    cols = np.repeat(np.arange(n), np.diff(out.indptr))  # its column
    out.data = out.data * f[rows] * f[cols]  # f_i * f_j is symmetric
    return out


def test_reuse_many_solves():
    """One build, many solves: each reused solve hits the residual target."""
    a = random_connected_graph(400, 0.02, seed=1)
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-8, seed=1))
    for k in range(6):
        b = mean_zero(np.random.default_rng(k), 400)
        x = solver.solve(b)
        assert _relres(a, x, b) < 1e-6


def test_solve_batch_matches_single_and_shape():
    """Batch solve equals per-system solves and returns a (B, n) array."""
    a = random_connected_graph(300, 0.03, seed=2)
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-9, seed=2))
    rng = np.random.default_rng(7)
    rhs = np.stack([mean_zero(rng, 300) for _ in range(8)])
    xb = solver.solve_batch(rhs)
    assert xb.shape == (8, 300)
    for i in range(8):
        xi = solver.solve(rhs[i])
        assert np.allclose(xb[i], xi, atol=1e-7, rtol=0.0)
        assert _relres(a, xb[i], rhs[i]) < 1e-6


def test_batch_parallel_speedup():
    """A large batch solves faster with threads than serially (GIL released)."""
    import time

    a = random_connected_graph(4000, 0.01, seed=3)
    rng = np.random.default_rng(3)
    rhs = np.stack([mean_zero(rng, 4000) for _ in range(16)])

    serial = LaplacianSolver(a, config=SolverConfig(tol=1e-8, seed=3, max_workers=1))
    parallel = LaplacianSolver(a, config=SolverConfig(tol=1e-8, seed=3, max_workers=None))
    serial.solve_batch(rhs)  # warm up
    parallel.solve_batch(rhs)

    t0 = time.perf_counter()
    serial.solve_batch(rhs)
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    parallel.solve_batch(rhs)
    t_par = time.perf_counter() - t0
    # Parallel must not be slower; on multicore it should be clearly faster.
    assert t_par <= t_serial * 1.1


def test_resolve_after_weight_update_without_rebuild():
    """update_weights re-solves the new matrix using the cached preconditioner."""
    a = random_connected_graph(350, 0.03, seed=4)
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-8, seed=4, rebuild_factor=100.0))
    ldli_before = solver._ldli  # the cached preconditioner object

    # A uniform weight scale leaves the preconditioner's quality unchanged
    # (eigenvalue ratios are invariant), so PCG converges WITHOUT a rebuild.
    a2 = a.copy()
    a2.data = a2.data * 1.1
    solver.update_weights(a2)
    b = mean_zero(np.random.default_rng(5), 350)
    x = solver.solve(b)
    assert _relres(a2, x, b) < 1e-6
    assert solver._ldli is ldli_before  # preconditioner reused, not rebuilt


def test_stale_preconditioner_triggers_rebuild_and_stays_correct():
    """A large weight change makes the stale preconditioner rebuild + stay correct."""
    a = random_connected_graph(350, 0.03, seed=14)
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-8, seed=14))
    a2 = _reweight(a, seed=99)  # large (10%) random reweight
    a2.data = a2.data * 5.0 + 0.5  # push it far from the cached preconditioner
    solver.update_weights(a2)
    b = mean_zero(np.random.default_rng(15), 350)
    x = solver.solve(b)
    assert _relres(a2, x, b) < 1e-6  # correct regardless of staleness (rebuild-retry)


def test_update_weights_rejects_pattern_change():
    """A pattern change is rejected (update_weights is values-only)."""
    a = random_connected_graph(60, 0.05, seed=6)
    solver = LaplacianSolver(a, config=SolverConfig(seed=6))
    denser = random_connected_graph(60, 0.2, seed=6)  # different pattern
    with pytest.raises(ValueError):
        solver.update_weights(denser)


def test_warm_start_matches_cold():
    """Warm-started and cold solves converge to the same solution."""
    a = random_connected_graph(300, 0.03, seed=8)
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-9, seed=8))
    b = mean_zero(np.random.default_rng(8), 300)
    x_cold = solver.solve(b)
    x_warm = solver.solve(b, x0=x_cold)
    assert np.allclose(x_cold, x_warm, atol=1e-7, rtol=0.0)


def test_forest_handle_is_exact():
    """A forest handle uses the exact fast path (no PCG) and is exact."""
    a = random_tree(200, seed=9)
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-10, seed=9))
    assert solver.is_forest
    b = mean_zero(np.random.default_rng(9), 200)
    x = solver.solve(b)
    assert _relres(a, x, b) < 1e-10


def test_purc_like_newton_sequence():
    """Mimic a PURC inner loop: build once, drift weights + RHS, warm-started re-solves."""
    a = grid_graph(30, 30, seed=10)
    n = a.shape[0]
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-8, seed=10))
    rng = np.random.default_rng(10)
    b = mean_zero(rng, n)
    x = np.zeros(n)
    current = a
    for _ in range(8):
        current = _reweight(current, seed=int(rng.integers(0, 1_000_000)))
        solver.update_weights(current)
        b = b + 0.02 * rng.standard_normal(n)
        b -= b.mean()
        x = solver.solve(b, x0=x)  # warm-start from previous iterate
        assert _relres(current, x, b) < 1e-6
