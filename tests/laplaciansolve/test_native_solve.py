"""
Stage 1b parity tests: native C++ solve path vs the Stage-0 reference.

Given an identical approxChol ``LDLinv`` (built by the Python reference), the C++
``ldl_solve`` and ``pcg_lap`` must reproduce the reference results to
floating-point round-off, and PCG must take the same iteration count (the solve
phase is deterministic and consumes no randomness).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import grid_graph, mean_zero, random_connected_graph

from purc.laplaciansolve.reference import approx_chol, approxchol_lap_pcg, lap, ldl_solve

core = pytest.importorskip(
    "purc.laplaciansolve._laplaciansolve_core",
    reason="native core not built (run: pip install --no-build-isolation -e .)",
)


def _csc_lap_int64(a):
    """
    Return ``lap(a)`` as CSC with int64 indptr/indices and sorted indices.

    Args:
        a: Adjacency matrix.

    Returns:
        The Laplacian as a sorted int64 CSC matrix.

    """
    la = lap(a).tocsc()
    la.sort_indices()
    la.indptr = la.indptr.astype(np.int64)
    la.indices = la.indices.astype(np.int64)
    return la


@pytest.mark.parametrize("n,p,seed", [(50, 0.1, 1), (200, 0.03, 2), (400, 0.02, 3)])
def test_native_ldl_solve_matches_reference(n, p, seed):
    """C++ ldl_solve == Python ldl_solve on identical LDLinv arrays."""
    a = random_connected_graph(n, p, seed=seed)
    ldli = approx_chol(a, rng=seed)
    b = mean_zero(np.random.default_rng(seed + 100), n)

    x_ref = ldl_solve(ldli, b, subtract_mean=True)
    x_cc = np.zeros(n, dtype=np.float64)
    core.ldl_solve_f64(ldli.col, ldli.colptr, ldli.rowval, ldli.fval, ldli.d, b, x_cc, True)

    assert np.allclose(x_cc, x_ref, atol=1e-11, rtol=0.0)


def test_native_ldl_solve_no_mean_subtraction():
    """The subtract_mean=False path matches the reference (SDDM use)."""
    a = random_connected_graph(80, 0.05, seed=7)
    ldli = approx_chol(a, rng=7)
    b = mean_zero(np.random.default_rng(8), 80)
    x_ref = ldl_solve(ldli, b, subtract_mean=False)
    x_cc = np.zeros(80, dtype=np.float64)
    core.ldl_solve_f64(ldli.col, ldli.colptr, ldli.rowval, ldli.fval, ldli.d, b, x_cc, False)
    assert np.allclose(x_cc, x_ref, atol=1e-11, rtol=0.0)


@pytest.mark.parametrize("n,p,seed", [(80, 0.05, 1), (300, 0.02, 2)])
def test_native_pcg_matches_reference(n, p, seed):
    """C++ PCG reproduces the reference iteration count and solution."""
    a = random_connected_graph(n, p, seed=seed)
    ldli = approx_chol(a, rng=seed)
    b = mean_zero(np.random.default_rng(seed + 9), n)

    la = _csc_lap_int64(a)
    bz = b - b.mean()
    x_cc = np.zeros(n, dtype=np.float64)
    its, relres, conv = core.pcg_lap_f64(
        la.indptr,
        la.indices,
        la.data,
        bz,
        ldli.col,
        ldli.colptr,
        ldli.rowval,
        ldli.fval,
        ldli.d,
        x_cc,
        1e-8,
        1000,
        5,
    )
    ref = approxchol_lap_pcg(a, b, tol=1e-8, seed=seed)

    assert conv
    assert its == ref.iterations  # deterministic solve phase => identical count
    assert np.allclose(x_cc, ref.x, atol=1e-9, rtol=0.0)
    assert np.linalg.norm(la @ x_cc - bz) / np.linalg.norm(bz) < 1e-6


def test_native_pcg_warm_start_reduces_iters():
    """Warm-starting PCG from the solution converges in zero/one iteration."""
    a = grid_graph(15, 15, seed=4)
    n = a.shape[0]
    ldli = approx_chol(a, rng=4)
    b = mean_zero(np.random.default_rng(4), n)
    la = _csc_lap_int64(a)
    bz = b - b.mean()

    cold = np.zeros(n, dtype=np.float64)
    its_cold, _, _ = core.pcg_lap_f64(
        la.indptr,
        la.indices,
        la.data,
        bz,
        ldli.col,
        ldli.colptr,
        ldli.rowval,
        ldli.fval,
        ldli.d,
        cold,
        1e-8,
        1000,
        5,
    )
    warm = cold.copy()  # start from the converged solution
    its_warm, _, conv = core.pcg_lap_f64(
        la.indptr,
        la.indices,
        la.data,
        bz,
        ldli.col,
        ldli.colptr,
        ldli.rowval,
        ldli.fval,
        ldli.d,
        warm,
        1e-8,
        1000,
        5,
    )
    assert conv
    assert its_warm <= its_cold
