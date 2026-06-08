"""
Stage 2 tests: PyTorch (CPU) support for the reusable handle and dispatch.

Verifies that torch tensors flow through the solver with the same results as
numpy (within dtype tolerance), that dtype/device are preserved on output, that
torch sparse / dense adjacencies are accepted, and that the one-shot dispatch
works across backends.  CUDA tensors must route to the (not-yet-implemented)
CUDA backend rather than silently falling back to CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import mean_zero, random_connected_graph, tolerance

pytest.importorskip(
    "purc.laplaciansolve._laplaciansolve_core",
    reason="native core not built (run: pip install --no-build-isolation -e .)",
)
torch = pytest.importorskip("torch")

import purc.laplaciansolve as ls  # noqa: E402
from purc.laplaciansolve import LaplacianSolver, SolverConfig  # noqa: E402
from purc.laplaciansolve.reference import lap  # noqa: E402


@pytest.fixture(params=[torch.float64, torch.float32], ids=["f64", "f32"])
def tdtype(request):
    """
    Torch dtype to test (float64 / float32).

    Args:
        request: pytest fixture request.

    Returns:
        The torch dtype.

    """
    return request.param


def test_torch_solve_matches_numpy(tdtype):
    """A torch-CPU solve matches the numpy solve and preserves dtype/device."""
    a = random_connected_graph(300, 0.03, seed=1)
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-9, seed=1))
    b = mean_zero(np.random.default_rng(1), 300)

    x_np = solver.solve(b)
    x_t = solver.solve(torch.tensor(b, dtype=tdtype))

    assert isinstance(x_t, torch.Tensor)
    assert x_t.dtype == tdtype
    assert x_t.device.type == "cpu"
    atol, rtol = tolerance(np.float32 if tdtype == torch.float32 else np.float64)
    assert np.allclose(x_t.to(torch.float64).numpy(), x_np, atol=atol, rtol=rtol)


def test_torch_batch_matches_numpy():
    """Torch batched solve matches numpy and returns a torch (B, n) tensor."""
    a = random_connected_graph(250, 0.03, seed=2)
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-9, seed=2))
    rng = np.random.default_rng(2)
    rhs = np.stack([mean_zero(rng, 250) for _ in range(6)])

    x_np = solver.solve_batch(rhs)
    x_t = solver.solve_batch(torch.tensor(rhs, dtype=torch.float64))
    assert isinstance(x_t, torch.Tensor)
    assert x_t.shape == (6, 250)
    assert np.allclose(x_t.numpy(), x_np, atol=1e-9, rtol=0.0)


def test_torch_residual():
    """The torch solution satisfies the Laplacian residual target."""
    a = random_connected_graph(400, 0.02, seed=3)
    solver = LaplacianSolver(a, config=SolverConfig(tol=1e-8, seed=3))
    b = torch.tensor(mean_zero(np.random.default_rng(3), 400), dtype=torch.float64)
    x = solver.solve(b)
    la = lap(a)
    bz = b.numpy() - b.numpy().mean()
    assert np.linalg.norm(la @ x.numpy() - bz) / np.linalg.norm(bz) < 1e-6


def test_handle_from_torch_dense_adjacency():
    """A handle can be built from a dense torch adjacency."""
    a = random_connected_graph(80, 0.06, seed=4)
    a_torch = torch.tensor(a.toarray(), dtype=torch.float64)
    solver = LaplacianSolver(a_torch, config=SolverConfig(tol=1e-9, seed=4))
    b = mean_zero(np.random.default_rng(4), 80)
    x = solver.solve(b)
    assert np.linalg.norm(lap(a) @ x - (b - b.mean())) / np.linalg.norm(b - b.mean()) < 1e-6


def test_handle_from_torch_sparse_adjacency():
    """A handle can be built from a sparse-COO torch adjacency."""
    a = random_connected_graph(80, 0.06, seed=5).tocoo()
    idx = torch.tensor(np.vstack([a.row, a.col]), dtype=torch.int64)
    val = torch.tensor(a.data, dtype=torch.float64)
    a_torch = torch.sparse_coo_tensor(idx, val, size=a.shape).coalesce()
    solver = LaplacianSolver(a_torch, config=SolverConfig(tol=1e-9, seed=5))
    b = mean_zero(np.random.default_rng(5), 80)
    x = solver.solve(b)
    a_csc = a.tocsc()
    assert np.linalg.norm(lap(a_csc) @ x - (b - b.mean())) / np.linalg.norm(b - b.mean()) < 1e-6


def test_dispatch_solve_numpy_and_torch():
    """The one-shot ls.solve works for numpy (1-D) and torch (2-D batch)."""
    a = random_connected_graph(150, 0.04, seed=6)
    b = mean_zero(np.random.default_rng(6), 150)
    x_np = ls.solve(a, b, tol=1e-8, seed=6)
    assert isinstance(x_np, np.ndarray)
    assert np.linalg.norm(lap(a) @ x_np - (b - b.mean())) / np.linalg.norm(b - b.mean()) < 1e-6

    rhs = torch.tensor(np.stack([b, mean_zero(np.random.default_rng(7), 150)]), dtype=torch.float64)
    x_t = ls.solve(a, rhs, tol=1e-8, seed=6)
    assert isinstance(x_t, torch.Tensor) and x_t.shape == (2, 150)
