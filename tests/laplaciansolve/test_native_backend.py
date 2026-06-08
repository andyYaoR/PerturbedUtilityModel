"""
Stage 1d tests: the native CPU backend (full approxchol_lap / approxchol_sddm).

Exercises the end-to-end native path - approxChol build + PCG solve, the SDDM
grounding embedding, connected-component splitting, and the exact forest fast
path - against residual targets and an exact dense solve.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    disconnected_graph,
    mean_zero,
    random_connected_graph,
    random_tree,
    sddm_from_adjacency,
)
from scipy import sparse

pytest.importorskip(
    "purc.laplaciansolve._laplaciansolve_core",
    reason="native core not built (run: pip install --no-build-isolation -e .)",
)

from purc.laplaciansolve.backends import native_cpu  # noqa: E402
from purc.laplaciansolve.reference import lap  # noqa: E402


@pytest.mark.parametrize("n,p,seed", [(60, 0.08, 1), (250, 0.025, 2), (500, 0.012, 3)])
def test_native_lap_residual(n, p, seed):
    """Native Laplacian solve reaches the residual target on connected graphs."""
    a = random_connected_graph(n, p, seed=seed)
    solve = native_cpu.approxchol_lap(a, tol=1e-8, maxits=1000, seed=seed)
    b = mean_zero(np.random.default_rng(seed + 5), n)
    x = solve(b)
    la = lap(a)
    assert np.linalg.norm(la @ x - b) / np.linalg.norm(b) < 1e-6


@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-6])
def test_native_sddm_vs_dense(eps):
    """Native SDDM solve matches an exact dense solve of M = lap(a) + eps I."""
    a = random_connected_graph(120, 0.05, seed=3)
    m = sddm_from_adjacency(a, eps)
    b = np.random.default_rng(4).standard_normal(120)
    x = native_cpu.approxchol_sddm(m, tol=1e-10, maxits=3000, seed=7)(b)
    x_exact = np.linalg.solve(m.toarray(), b)
    assert np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact) < 1e-4
    assert np.linalg.norm(m @ x - b) / np.linalg.norm(b) < 1e-7


def test_native_forest_fast_path_is_exact():
    """On a forest the native solve is exact (factorization needs no PCG)."""
    a = random_tree(256, seed=5)
    solve = native_cpu.approxchol_lap(a, tol=1e-10, seed=5)
    b = mean_zero(np.random.default_rng(6), 256)
    x = solve(b)
    la = lap(a)
    assert np.linalg.norm(la @ x - b) / np.linalg.norm(b) < 1e-10


def test_native_disconnected_graph():
    """Disconnected graphs solve per component; singletons return zeros."""
    a, _ = disconnected_graph(seed=0)
    n = a.shape[0]
    solve = native_cpu.approxchol_lap(a, tol=1e-10, seed=1)
    b = np.zeros(n)
    b[0:3] = mean_zero(np.random.default_rng(1), 3)
    b[3:6] = mean_zero(np.random.default_rng(2), 3)
    x = solve(b)
    la = lap(a)
    assert np.linalg.norm((la @ x - b)[0:3]) < 1e-6
    assert np.linalg.norm((la @ x - b)[3:6]) < 1e-6
    assert x[6] == 0.0


def test_native_seed_reproducible():
    """The same seed yields the same native solution."""
    a = random_connected_graph(100, 0.05, seed=9)
    b = mean_zero(np.random.default_rng(9), 100)
    x1 = native_cpu.approxchol_lap(a, tol=1e-8, seed=42)(b)
    x2 = native_cpu.approxchol_lap(a, tol=1e-8, seed=42)(b)
    assert np.array_equal(x1, x2)
