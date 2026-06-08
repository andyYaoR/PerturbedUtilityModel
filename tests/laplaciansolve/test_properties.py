"""
Property-based tests for the reference solver (Hypothesis).

These assert solver invariants over many randomly generated graphs rather than a
handful of fixed instances: the Laplacian solve recovers the RHS on the
non-kernel subspace, the SDDM solve is exact, and refilling-then-solving matches
solving the original (a precursor of the Stage-3 caching invariant).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sddm_from_adjacency
from scipy import sparse

from purc.laplaciansolve.reference import approxchol_lap, approxchol_sddm, lap

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


def _random_connected(n: int, extra_edges: int, weight_seed: int) -> sparse.csc_matrix:
    """
    Build a connected weighted graph: a spanning path plus random chords.

    Args:
        n: Number of vertices.
        extra_edges: Number of additional random edges.
        weight_seed: RNG seed for edge weights and chord endpoints.

    Returns:
        A connected, symmetric adjacency (CSC).

    """
    r = np.random.default_rng(weight_seed)
    rows, cols, vals = [], [], []

    def add(u: int, v: int) -> None:
        if u == v:
            return
        w = float(r.random() + 0.1)
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([w, w])

    for i in range(n - 1):  # spanning path => connected
        add(i, i + 1)
    for _ in range(extra_edges):
        add(int(r.integers(0, n)), int(r.integers(0, n)))
    return sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=2, max_value=40),
    extra=st.integers(min_value=0, max_value=30),
    seed=st.integers(min_value=0, max_value=10_000),
    rhs_seed=st.integers(min_value=0, max_value=10_000),
)
def test_lap_solve_recovers_rhs(n, extra, seed, rhs_seed):
    """lap(a) @ solve(b) recovers b on the mean-zero subspace."""
    a = _random_connected(n, extra, seed)
    b = np.random.default_rng(rhs_seed).standard_normal(n)
    b -= b.mean()
    if np.linalg.norm(b) < 1e-12:
        return
    x = approxchol_lap(a, tol=1e-9, maxits=5000, seed=seed)(b)
    residual = lap(a) @ x - b
    assert np.linalg.norm(residual) / np.linalg.norm(b) < 1e-5


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=2, max_value=40),
    extra=st.integers(min_value=0, max_value=30),
    seed=st.integers(min_value=0, max_value=10_000),
    log_eps=st.integers(min_value=-8, max_value=0),
)
def test_sddm_solve_is_exact(n, extra, seed, log_eps):
    """(lap(a) + eps I) @ solve(b) recovers a general b."""
    a = _random_connected(n, extra, seed)
    eps = 10.0**log_eps
    m = sddm_from_adjacency(a, eps)
    b = np.random.default_rng(seed + 1).standard_normal(n)
    x = approxchol_sddm(m, tol=1e-10, maxits=10_000, seed=seed)(b)
    assert np.linalg.norm(m @ x - b) / np.linalg.norm(b) < 1e-5
