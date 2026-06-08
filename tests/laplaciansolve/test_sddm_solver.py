"""
Stage 3 tests: the reusable SDDMSolver handle + the SiouxFalls example.

Covers the PURC matrix M = H + eps*I: exactness vs a dense solve, batched solve,
re-solve after a weight/regularizer update, torch I/O, and an end-to-end check
on the SiouxFalls network with a quadratic perturbation (the example's loader).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
from conftest import random_connected_graph
from scipy import sparse

pytest.importorskip(
    "purc.laplaciansolve._laplaciansolve_core",
    reason="native core not built (run: pip install --no-build-isolation -e .)",
)

from purc.laplaciansolve import SDDMSolver, SolverConfig  # noqa: E402
from purc.laplaciansolve.reference import lap  # noqa: E402


def _sddm(a, eps):
    """
    Assemble M = lap(a) + eps*I as CSC.

    Args:
        a: Adjacency matrix.
        eps: Positive regularizer.

    Returns:
        The SDDM matrix as CSC.

    """
    return sparse.csc_matrix(lap(a) + eps * sparse.eye(a.shape[0]))


@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-6])
def test_sddm_matches_dense(eps):
    """SDDMSolver matches an exact dense solve of M = lap(a) + eps I."""
    a = random_connected_graph(150, 0.04, seed=1)
    m = _sddm(a, eps)
    solver = SDDMSolver(m, config=SolverConfig(tol=1e-10, maxits=3000, seed=1))
    b = np.random.default_rng(1).standard_normal(150)
    x = solver.solve(b)
    x_exact = np.linalg.solve(m.toarray(), b)
    assert np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact) < 1e-4
    assert np.linalg.norm(m @ x - b) / np.linalg.norm(b) < 1e-7


def test_sddm_batch_matches_single():
    """Batched SDDM solve equals per-system solves and the dense solution."""
    a = random_connected_graph(120, 0.05, seed=2)
    m = _sddm(a, 1e-3)
    solver = SDDMSolver(m, config=SolverConfig(tol=1e-10, maxits=3000, seed=2))
    rng = np.random.default_rng(2)
    rhs = rng.standard_normal((7, 120))
    xb = solver.solve_batch(rhs)
    assert xb.shape == (7, 120)
    for i in range(7):
        assert np.allclose(xb[i], solver.solve(rhs[i]), atol=1e-7, rtol=0.0)
        assert np.linalg.norm(m @ xb[i] - rhs[i]) / np.linalg.norm(rhs[i]) < 1e-6


def test_sddm_update_resolves_correctly():
    """After update() with a new regularizer, solves target the new matrix."""
    a = random_connected_graph(140, 0.04, seed=3)
    solver = SDDMSolver(_sddm(a, 1e-3), config=SolverConfig(tol=1e-10, maxits=3000, seed=3))
    m2 = _sddm(a, 5e-2)
    solver.update(m2)
    b = np.random.default_rng(3).standard_normal(140)
    x = solver.solve(b)
    assert np.linalg.norm(m2 @ x - b) / np.linalg.norm(b) < 1e-7


def test_sddm_torch_io():
    """SDDMSolver accepts/returns torch tensors (CPU)."""
    torch = pytest.importorskip("torch")
    a = random_connected_graph(100, 0.05, seed=4)
    m = _sddm(a, 1e-3)
    solver = SDDMSolver(m, config=SolverConfig(tol=1e-10, maxits=3000, seed=4))
    b = np.random.default_rng(4).standard_normal(100)
    x = solver.solve(torch.tensor(b, dtype=torch.float64))
    assert isinstance(x, torch.Tensor)
    assert np.linalg.norm(m @ x.numpy() - b) / np.linalg.norm(b) < 1e-7


def test_sddm_auto_selects_cholmod_when_available():
    """method='auto' picks the direct CHOLMOD backend when it is built."""
    from purc.laplaciansolve._loader import has_cholmod

    a = random_connected_graph(60, 0.06, seed=20)
    s = SDDMSolver(_sddm(a, 1e-3))
    assert s.method == ("cholmod" if has_cholmod() else "approxchol")


@pytest.mark.parametrize("method", ["cholmod", "approxchol"])
def test_sddm_both_backends_match_dense(method):
    """Both backends ('cholmod' direct and 'approxchol' iterative) match dense."""
    if method == "cholmod":
        pytest.importorskip("purc.laplaciansolve._laplaciansolve_cholmod")
    a = random_connected_graph(120, 0.05, seed=21)
    m = _sddm(a, 1e-3)
    s = SDDMSolver(m, config=SolverConfig(tol=1e-10, maxits=3000, seed=21, method=method))
    assert s.method == method
    b = np.random.default_rng(21).standard_normal(120)
    x = s.solve(b)
    assert np.linalg.norm(x - np.linalg.solve(m.toarray(), b)) / np.linalg.norm(
        np.linalg.solve(m.toarray(), b)
    ) < 1e-4


def test_cholmod_update_and_batch():
    """CHOLMOD backend: numeric update (new eps) + batched solve stay exact."""
    pytest.importorskip("purc.laplaciansolve._laplaciansolve_cholmod")
    a = random_connected_graph(150, 0.04, seed=22)
    s = SDDMSolver(_sddm(a, 1e-3), config=SolverConfig(method="cholmod"))
    m2 = _sddm(a, 1e-1)
    s.update(m2)  # numeric refactorization, same pattern
    rng = np.random.default_rng(22)
    rhs = rng.standard_normal((6, 150))
    xb = s.solve_batch(rhs)
    assert xb.shape == (6, 150)
    for i in range(6):
        assert np.linalg.norm(m2 @ xb[i] - rhs[i]) / np.linalg.norm(rhs[i]) < 1e-9


def _load_example():
    """
    Import the SiouxFalls example module by path.

    Returns:
        The imported example module.

    """
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "examples", "siouxfalls_laplacian_solve.py"
    )
    spec = importlib.util.spec_from_file_location("sf_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_siouxfalls_quadratic_laplacian_solve():
    """End-to-end: SiouxFalls quadratic-perturbation Newton matrix solves exactly."""
    sf = _load_example()
    n_nodes, links = sf.load_tntp_links(sf._DATA)
    assert n_nodes == 24
    assert len(links) == 76  # directed links

    adjacency = sf.build_weighted_laplacian_adjacency(n_nodes, links)
    assert adjacency.shape == (24, 24)
    assert abs(adjacency - adjacency.T).nnz == 0  # symmetric

    m = sparse.csc_matrix(lap(adjacency) + 1e-6 * sparse.eye(24))
    solver = SDDMSolver(m, config=SolverConfig(tol=1e-10, maxits=2000, seed=0))
    b = np.random.default_rng(0).standard_normal(24)
    x = solver.solve(b)
    x_dense = np.linalg.solve(m.toarray(), b)
    assert np.linalg.norm(m @ x - b) / np.linalg.norm(b) < 1e-7
    # Relative check: eps=1e-6 makes M near-singular (cond ~1e6), so an absolute
    # tolerance is inappropriate even though the relative accuracy is excellent.
    assert np.linalg.norm(x - x_dense) / np.linalg.norm(x_dense) < 1e-5
