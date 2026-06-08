"""
Stage 1c parity tests: native C++ approxChol build vs the Stage-0 reference.

Driven by an identical injected sample stream, the C++ ``approx_chol_f64`` build
must reproduce the reference ``LDLinv`` *bit-for-bit* (the elimination order,
sparsity, multipliers, and diagonal are all deterministic given the RNG).  An
end-to-end build+solve in C++ is also checked against the Laplacian residual.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import grid_graph, mean_zero, random_connected_graph, random_tree

from purc.laplaciansolve.reference import ArrayStream, approx_chol, lap

core = pytest.importorskip(
    "purc.laplaciansolve._laplaciansolve_core",
    reason="native core not built (run: pip install --no-build-isolation -e .)",
)


def _csc_int64(a):
    """
    Return *a* as a sorted CSC with int64 index arrays.

    Args:
        a: Adjacency matrix.

    Returns:
        ``(indptr, indices, data)`` ready for the native build.

    """
    a = a.tocsc()
    a.sort_indices()
    return (
        a.indptr.astype(np.int64),
        a.indices.astype(np.int64),
        a.data.astype(np.float64),
    )


def _graphs():
    """
    Yield (label, adjacency) test graphs spanning the relevant regimes.

    Yields:
        Pairs of ``(id, csc_adjacency)``.

    """
    yield "rand_50", random_connected_graph(50, 0.1, seed=1)
    yield "rand_200", random_connected_graph(200, 0.04, seed=2)
    yield "rand_400", random_connected_graph(400, 0.02, seed=3)
    yield "grid_15x15", grid_graph(15, 15, seed=4)
    yield "tree_128", random_tree(128, seed=5)


_CASES = list(_graphs())


@pytest.mark.parametrize("label,a", _CASES, ids=[c[0] for c in _CASES])
def test_native_build_bit_exact(label, a):
    """C++ approx_chol == reference approx_chol, bit-for-bit, on shared samples."""
    indptr, indices, data = _csc_int64(a)
    nnz = int(indptr[-1])
    samples = np.random.default_rng(hash(label) % 2**32).random(20 * max(nnz, 1))

    ref = approx_chol(a, rng=ArrayStream(samples.copy()))
    col, colptr, rowval, fval, d = core.approx_chol_f64(indptr, indices, data, samples)

    assert np.array_equal(col, ref.col)
    assert np.array_equal(colptr, ref.colptr)
    assert np.array_equal(rowval, ref.rowval)
    assert np.array_equal(fval, ref.fval)  # bit-exact multipliers
    assert np.array_equal(d, ref.d)  # bit-exact diagonal


def test_native_build_then_solve_residual():
    """An entirely native build + solve reaches the Laplacian residual target."""
    a = random_connected_graph(300, 0.03, seed=7)
    indptr, indices, data = _csc_int64(a)
    samples = np.random.default_rng(7).random(20 * int(indptr[-1]))

    col, colptr, rowval, fval, d = core.approx_chol_f64(indptr, indices, data, samples)

    la = lap(a).tocsc()
    la.sort_indices()
    b = mean_zero(np.random.default_rng(8), 300)
    bz = b - b.mean()
    x = np.zeros(300, dtype=np.float64)
    its, relres, conv = core.pcg_lap_f64(
        la.indptr.astype(np.int64),
        la.indices.astype(np.int64),
        la.data,
        bz,
        col,
        colptr,
        rowval,
        fval,
        d,
        x,
        1e-8,
        1000,
        5,
    )
    assert conv
    assert np.linalg.norm(la @ x - bz) / np.linalg.norm(bz) < 1e-6


def test_native_build_forest_uses_no_samples():
    """A forest build consumes no randomness (leaves never sample)."""
    a = random_tree(200, seed=11)
    indptr, indices, data = _csc_int64(a)
    # Pass an empty sample stream: a forest must not request any draws.
    empty = np.zeros(0, dtype=np.float64)
    col, colptr, rowval, fval, d = core.approx_chol_f64(indptr, indices, data, empty)
    ref = approx_chol(a, rng=ArrayStream(empty.copy()))
    assert np.array_equal(fval, ref.fval)
    assert np.array_equal(d, ref.d)
