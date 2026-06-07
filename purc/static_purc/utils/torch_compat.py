"""
Torch-first array handling with zero-copy bridges to NumPy / SciPy.

PURCSolver is torch-native: tensors are the primary data type for the public API
and all solver math, defaulting to CPU ``float64`` but written device-agnostically
(the heavy Newton solve is delegated to LaplacianSolve, which is CPU, so tensors
round-trip to CPU only for that call).  A handful of dependencies are NumPy/SciPy
based (the CSC pattern build, the cvxpy oracle, feasibility checks); these are
bridged with :func:`to_numpy` / :func:`as_tensor`, which share memory with the
torch buffer on CPU (``tensor.numpy()`` / ``torch.from_numpy``) so the transfer is
zero-copy and low-overhead.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

# The package default: CPU float64 (matches the CHOLMOD solve and the estimation
# layer).  Callers may pass tensors on other devices/dtypes; ops are written to
# follow the input where possible and only force CPU for the linear solve.
DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = torch.device("cpu")

TensorLike = Union[torch.Tensor, np.ndarray, float, int, list, tuple]


def as_tensor(
    x: TensorLike,
    *,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Coerce ``x`` to a torch tensor, zero-copy from NumPy on CPU when possible.

    Args:
        x: A tensor, numpy array, scalar, or (nested) sequence.
        dtype: Target dtype (default ``float64``).
        device: Target device; ``None`` keeps an existing tensor's device or uses
            CPU for non-tensor inputs.

    Returns:
        A torch tensor of the requested dtype/device.

    """
    if isinstance(x, torch.Tensor):
        out = x
        if device is not None and out.device != torch.device(device):
            out = out.to(device)
        if out.dtype != dtype:
            out = out.to(dtype)
        return out
    if isinstance(x, np.ndarray):
        # torch.from_numpy shares memory; .to(dtype) copies only on dtype change.
        t = torch.from_numpy(np.ascontiguousarray(x))
        t = t.to(dtype)
        return t if device is None else t.to(device)
    return torch.as_tensor(x, dtype=dtype, device=device or DEFAULT_DEVICE)


def to_numpy(x: TensorLike) -> np.ndarray:
    """
    Return a NumPy view/array of ``x``, zero-copy from a CPU torch tensor.

    Args:
        x: A torch tensor (any device) or array-like.

    Returns:
        A contiguous ``numpy.ndarray`` (shares memory with a CPU tensor).

    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Return ``x`` as a 1-D tensor (ravel), preserving dtype/device.

    Args:
        x: A torch tensor.

    Returns:
        The flattened tensor.

    """
    return x.reshape(-1)
