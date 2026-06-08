"""
Shared array-backend plumbing for native direct SDDM backends.

Both the CHOLMOD and the forest-LDL handles wrap a native solver object exposing
``solve(b, x_out, k)`` (k right-hand sides, flat ``[k, n]`` row-major buffers) and
``update(data)`` (numeric refactorization on a fixed pattern).  This base class
provides the numpy / scipy / torch I/O, the contiguous-float64 fast path, and the
fixed-pattern caching; subclasses only build ``self._solver`` in their
constructor after calling :meth:`_setup_pattern`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._convert import VectorAdapter, adjacency_to_csc


def _is_contig_f64(arr, ndim: int) -> bool:
    """
    Return whether *arr* is a C-contiguous float64 numpy array of rank *ndim*.

    Args:
        arr: The candidate right-hand side.
        ndim: The required number of dimensions (1 for a vector, 2 for a batch).

    Returns:
        ``True`` if *arr* can go straight to the native solve with no conversion.

    """
    return (
        type(arr) is np.ndarray
        and arr.dtype == np.float64
        and arr.ndim == ndim
        and arr.flags.c_contiguous
    )


class _DirectSDDMBase:
    """Common I/O for a native direct solver with ``solve``/``update`` methods."""

    _solver = None  # the native handle; set by the subclass constructor

    def _setup_pattern(self, m):
        """
        Convert *m* to a sorted float64 CSC and cache its sparsity pattern.

        Args:
            m: A symmetric matrix (scipy / numpy / torch; CPU).

        Returns:
            The matrix as a sorted float64 :class:`scipy.sparse.csc_matrix`.

        Raises:
            ValueError: If *m* is not square.

        """
        csc = adjacency_to_csc(m)
        csc.sort_indices()
        if csc.shape[0] != csc.shape[1]:
            raise ValueError(f"matrix must be square, got {csc.shape}")
        self._n = int(csc.shape[0])
        self._indptr = csc.indptr.astype(np.int64)
        self._indices = csc.indices.astype(np.int64)
        self._nnz = int(csc.nnz)
        return csc

    @property
    def n(self) -> int:
        """
        Dimension of the system.

        Returns:
            The matrix dimension.

        """
        return self._n

    def update(self, m) -> None:
        """
        Numeric refactorization with new values on the cached pattern.

        Args:
            m: Updated matrix sharing the original sparsity pattern.

        Raises:
            ValueError: If the sparsity pattern differs from the cached one.

        """
        csc = adjacency_to_csc(m)
        csc.sort_indices()
        if csc.nnz != self._nnz or not np.array_equal(csc.indices.astype(np.int64), self._indices):
            raise ValueError("update requires the same sparsity pattern")
        self._solver.update(csc.data.astype(np.float64))

    def solve(self, b, *, x0: Optional[object] = None):
        """
        Solve ``M x = b`` for a single right-hand side.

        Args:
            b: Right-hand side of length ``n`` (numpy / scipy / torch).
            x0: Ignored (a direct solve needs no warm start; kept for interface).

        Returns:
            The solution in the same array kind / dtype / device as ``b``.

        """
        if _is_contig_f64(b, 1):
            x = np.empty(self._n, dtype=np.float64)
            self._solver.solve(b, x, 1)
            return x
        adapter = VectorAdapter(b)
        bz = np.ascontiguousarray(adapter.array, dtype=np.float64)
        x = np.empty(self._n, dtype=np.float64)
        self._solver.solve(bz, x, 1)
        return adapter.restore(x)

    def solve_batch(self, rhs, *, x0: Optional[object] = None):
        """
        Solve ``M x = b`` for a batch of right-hand sides in one native call.

        Args:
            rhs: A ``(B, n)`` array or list of length-``n`` vectors
                (numpy / scipy / torch).
            x0: Ignored (direct solve).

        Returns:
            A ``(B, n)`` solution in the same array kind as ``rhs``.

        """
        if _is_contig_f64(rhs, 2):
            batch = rhs.shape[0]
            out = np.empty(batch * self._n, dtype=np.float64)
            self._solver.solve(rhs.reshape(-1), out, batch)
            return out.reshape(batch, self._n)
        adapter = VectorAdapter(rhs)
        r = np.ascontiguousarray(adapter.array, dtype=np.float64)
        if r.ndim == 1:
            r = r[None, :]
        batch = r.shape[0]
        out = np.empty(batch * self._n, dtype=np.float64)
        # [B, n] row-major == [n, B] column-major (what the native solve expects).
        self._solver.solve(r.reshape(-1), out, batch)
        return adapter.restore(out.reshape(batch, self._n))
