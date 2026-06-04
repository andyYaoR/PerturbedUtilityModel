"""
CSR sparse matrix-vector product for the solver hot path.

``A @ x`` and ``A^T @ lambda`` are evaluated several times per Newton iteration
(residual, dual objective, line search), so they go through a dedicated,
GIL-released native float64 CSR kernel when the native core is built, with an
exact SciPy fallback otherwise.  Inputs/outputs are NumPy on CPU (the torch
solver bridges its tensors zero-copy around these calls); the native path keeps
full float64 precision and avoids the overhead of torch's beta sparse-CSR matmul.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .native import native_available, native_core


class CSRMatVec:
    """
    A fixed CSR matrix exposing a fast ``matvec`` (native kernel or SciPy).

    Args:
        matrix: The matrix as a SciPy sparse / dense array; stored as CSR.

    """

    def __init__(self, matrix) -> None:
        csr = sp.csr_matrix(matrix).astype(np.float64)
        csr.sort_indices()
        self._scipy = csr
        self.nrows, self.ncols = csr.shape
        # int64 index arrays for the native kernel (scipy defaults to int32).
        self._indptr = csr.indptr.astype(np.int64)
        self._indices = csr.indices.astype(np.int64)
        self._data = np.ascontiguousarray(csr.data, dtype=np.float64)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Compute ``A @ x``.

        Args:
            x: Dense vector of length ``ncols`` (contiguous float64 on CPU).

        Returns:
            The dense product ``A @ x`` of length ``nrows``.

        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        if native_available():
            y = np.empty(self.nrows, dtype=np.float64)
            native_core().csr_spmv_f64(self._indptr, self._indices, self._data, x, y)
            return y
        return self._scipy @ x
