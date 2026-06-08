"""
CUDA scaffolding tests (Stage 1e) + GPU parity harness (activates on a GPU host).

On a CPU-only host these assert the infrastructure behaves correctly: the CUDA
backend reports unavailable and raises a clear NotImplementedError rather than
silently falling back.  The GPU parity cases skip unless a CUDA device and the
compiled CUDA module are both present; they go live once the Stage-4 kernels are
implemented.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import mean_zero, random_connected_graph

from purc.laplaciansolve.backends import torch_cuda

pytestmark = pytest.mark.cuda

requires_cuda = pytest.mark.skipif(
    not torch_cuda.cuda_available(),
    reason="no CUDA device + compiled CUDA module",
)


def test_cuda_backend_raises_when_pending():
    """The CUDA backend raises NotImplementedError (no silent fallback)."""
    a = random_connected_graph(20, 0.2, seed=1)
    with pytest.raises(NotImplementedError):
        torch_cuda.approxchol_lap(a)
    with pytest.raises(NotImplementedError):
        torch_cuda.approxchol_sddm(a)


def test_cuda_available_is_bool():
    """cuda_available() returns a bool without importing torch eagerly failing."""
    assert isinstance(torch_cuda.cuda_available(), bool)


@requires_cuda
def test_cuda_lap_matches_cpu():
    """GPU Laplacian solve matches the CPU backend (activates on a GPU host)."""
    from conftest import tolerance

    from purc.laplaciansolve.backends import native_cpu

    a = random_connected_graph(500, 0.01, seed=2)
    b = mean_zero(np.random.default_rng(2), 500)
    x_cpu = native_cpu.approxchol_lap(a, tol=1e-8, seed=2)(b)
    x_gpu = torch_cuda.approxchol_lap(a, tol=1e-8, seed=2)(b)
    atol, rtol = tolerance(np.float64)
    assert np.allclose(np.asarray(x_gpu), x_cpu, atol=atol, rtol=rtol)
