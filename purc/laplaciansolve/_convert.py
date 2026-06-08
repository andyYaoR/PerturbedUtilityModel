"""
Array-backend conversion (numpy / scipy / torch) for the solver boundary.

Keeps the native core numpy-only while letting callers pass numpy, scipy, or
torch (CPU) arrays and get results back in the same type/dtype/device.  CPU
torch tensors convert through the numpy buffer (no device copy); CUDA tensors
are rejected here (the CPU handle never silently falls back to host compute -
they belong on the CUDA backend, raised until implemented).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse

_TORCH_RESOLVED = False
_TORCH_MOD = None


def _torch():
    """
    Return the imported torch module, or ``None`` if torch is unavailable.

    The result is memoized after the first call so the hot solve path does not
    re-enter the import machinery on every invocation.

    Returns:
        The ``torch`` module or ``None``.

    """
    global _TORCH_RESOLVED, _TORCH_MOD
    if not _TORCH_RESOLVED:
        try:
            import torch

            _TORCH_MOD = torch
        except ImportError:  # pragma: no cover - torch is an optional dependency
            _TORCH_MOD = None
        _TORCH_RESOLVED = True
    return _TORCH_MOD


def adjacency_to_csc(a) -> sparse.csc_matrix:
    """
    Convert a numpy / scipy / torch adjacency to a float64 scipy CSC matrix.

    Args:
        a: Adjacency as a scipy sparse matrix, dense numpy array, or torch
            tensor (dense or sparse COO; CPU).

    Returns:
        The adjacency as a float64 :class:`scipy.sparse.csc_matrix`.

    Raises:
        NotImplementedError: If *a* is a CUDA tensor (use the CUDA backend).

    """
    torch = _torch()
    if torch is not None and isinstance(a, torch.Tensor):
        if a.is_cuda:
            raise NotImplementedError(
                "CUDA adjacency tensors require the CUDA backend (not yet "
                "implemented); move to CPU or see docs/CUDA.md."
            )
        a = a.detach().cpu()
        if a.is_sparse:
            a = a.coalesce()
            idx = a.indices().numpy()
            val = a.values().to(torch.float64).numpy()
            return sparse.csc_matrix((val, (idx[0], idx[1])), shape=tuple(a.shape))
        a = a.to(torch.float64).numpy()
    return sparse.csc_matrix(a, dtype=np.float64)


class VectorAdapter:
    """
    Captures an input vector/batch's array kind so results return in that kind.

    Exposes the input as a contiguous float64 numpy array for the native solve
    and restores solutions to the original type, dtype, and device.
    """

    def __init__(self, x) -> None:
        """
        Capture *x*'s backend and expose a float64 numpy view/copy.

        Args:
            x: A numpy array, array-like, or CPU torch tensor.

        Raises:
            NotImplementedError: If *x* is a CUDA torch tensor.

        """
        torch = _torch()
        if torch is not None and isinstance(x, torch.Tensor):
            if x.is_cuda:
                raise NotImplementedError(
                    "CUDA tensors require the CUDA backend (not yet implemented); "
                    "move to CPU or see docs/CUDA.md."
                )
            self.kind = "torch"
            self.dtype = x.dtype
            self.device = x.device
            self._np = x.detach().cpu().to(torch.float64).contiguous().numpy()
        else:
            arr = np.asarray(x)
            self.kind = "numpy"
            self.dtype = arr.dtype
            self.device = None
            self._np = np.ascontiguousarray(arr, dtype=np.float64)

    @property
    def array(self) -> np.ndarray:
        """
        The float64 numpy view of the input.

        Returns:
            A contiguous float64 array.

        """
        return self._np

    def restore(self, result: np.ndarray):
        """
        Convert a numpy result back to the input's type/dtype/device.

        Args:
            result: The float64 numpy solution.

        Returns:
            The solution as the original array kind (numpy or torch).

        """
        if self.kind == "torch":
            torch = _torch()
            tensor = torch.from_numpy(np.ascontiguousarray(result))
            return tensor.to(dtype=self.dtype, device=self.device)
        if np.dtype(self.dtype).kind == "f" and np.dtype(self.dtype) != np.float64:
            return result.astype(self.dtype, copy=False)
        return result


def is_torch(x) -> bool:
    """
    Return whether *x* is a torch tensor.

    Args:
        x: Any object.

    Returns:
        ``True`` if *x* is a ``torch.Tensor``.

    """
    torch = _torch()
    return torch is not None and isinstance(x, torch.Tensor)


def detect_backend(x) -> str:
    """
    Name the array backend of *x* (``"torch"``, ``"scipy"``, or ``"numpy"``).

    Args:
        x: An array, sparse matrix, or tensor.

    Returns:
        The backend name.

    """
    if is_torch(x):
        return "torch"
    if sparse.issparse(x):
        return "scipy"
    return "numpy"


def torch_available() -> bool:
    """
    Return whether torch is importable.

    Returns:
        ``True`` if torch can be imported.

    """
    return _torch() is not None
