"""
Torch CUDA backend (Stage 1e scaffolding; kernel bodies pending).

The CUDA build infrastructure, native module, and this dispatch surface exist
now so that a GPU machine only needs to implement the kernels in
``src/purc.laplaciansolve/cuda/`` (see ``docs/CUDA.md``).  Until then every entry
point raises :class:`NotImplementedError` - there is deliberately no silent
fallback to the CPU backend.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

_DOC = "see docs/CUDA.md"


def cuda_available() -> bool:
    """
    Report whether both a CUDA device and the compiled CUDA module are present.

    Returns:
        ``True`` only if PyTorch sees a CUDA device *and* the
        ``_laplaciansolve_cuda`` extension was built; otherwise ``False``.

    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        importlib.import_module("purc.laplaciansolve._laplaciansolve_cuda")
    except ImportError:
        return False
    return True


def _pending(name: str) -> None:
    """
    Raise a clear not-implemented error for a pending CUDA kernel.

    Args:
        name: The logical solver entry point being requested.

    Raises:
        NotImplementedError: Always - the GPU kernels are not implemented yet.

    """
    raise NotImplementedError(
        f"LaplacianSolve CUDA backend '{name}' is not implemented yet; "
        f"the build infrastructure is in place ({_DOC})."
    )


def approxchol_lap(a: Any, **kwargs: Any) -> Callable[[Any], Any]:
    """
    CUDA Laplacian solver (kernel bodies pending).

    Args:
        a: Adjacency matrix.
        **kwargs: Solver options (forwarded once implemented).

    Returns:
        A solver closure once the kernels are implemented.

    Raises:
        NotImplementedError: Until the GPU kernels are implemented.

    """
    _pending("approxchol_lap")


def approxchol_sddm(m: Any, **kwargs: Any) -> Callable[[Any], Any]:
    """
    CUDA SDDM solver (kernel bodies pending).

    Args:
        m: SDDM matrix.
        **kwargs: Solver options (forwarded once implemented).

    Returns:
        A solver closure once the kernels are implemented.

    Raises:
        NotImplementedError: Until the GPU kernels are implemented.

    """
    _pending("approxchol_sddm")
