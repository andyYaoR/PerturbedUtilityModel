"""
Native approxChol factorization handle.

Wraps the ``LDLinv`` arrays returned by the C++ build and exposes the
triangular solve, both backed by the native core.  This is the CPU counterpart
of the reference :class:`~purc.laplaciansolve.reference.ldl_solve.LDLinv`, used by the
:mod:`purc.laplaciansolve.backends.native_cpu` solvers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse

from ._loader import native_core


class NativeLDL:
    """
    An approxChol ``L D L^T`` factorization computed by the native core.

    Attributes:
        col: Elimination order (int64).
        colptr: Column pointers (int64).
        rowval: ``L`` row indices (int64).
        fval: ``L`` multipliers (float64).
        d: Diagonal of ``D`` (float64).

    """

    __slots__ = ("col", "colptr", "rowval", "fval", "d")

    def __init__(
        self,
        col: np.ndarray,
        colptr: np.ndarray,
        rowval: np.ndarray,
        fval: np.ndarray,
        d: np.ndarray,
    ) -> None:
        """
        Store the factor arrays.

        Args:
            col: Elimination order.
            colptr: Column pointers.
            rowval: ``L`` row indices.
            fval: ``L`` multipliers.
            d: Diagonal of ``D``.

        """
        self.col = col
        self.colptr = colptr
        self.rowval = rowval
        self.fval = fval
        self.d = d

    @property
    def n(self) -> int:
        """
        Dimension of the factorized matrix.

        Returns:
            The number of vertices ``n``.

        """
        return int(self.d.shape[0])

    def solve(self, b: np.ndarray, *, subtract_mean: bool = True) -> np.ndarray:
        """
        Apply the factorization: ``x = (L D L^T)^{-1} b``.

        Args:
            b: Right-hand side of length ``n``.
            subtract_mean: Project out the constant null space (Laplacian case).

        Returns:
            The solution vector.

        """
        b = np.ascontiguousarray(b, dtype=np.float64)
        x = np.zeros(self.n, dtype=np.float64)
        native_core().ldl_solve_f64(
            self.col, self.colptr, self.rowval, self.fval, self.d, b, x, subtract_mean
        )
        return x


def build_approxchol(
    a: sparse.csc_matrix,
    *,
    seed: int = 0,
    samples: Optional[np.ndarray] = None,
) -> NativeLDL:
    """
    Build a :class:`NativeLDL` approxChol factorization of ``lap(a)``.

    Args:
        a: Symmetric, nonnegative, zero-diagonal adjacency matrix (CSC).
        seed: Seed for the internal RNG (used when ``samples`` is ``None``).
        samples: Optional explicit uniform ``[0, 1)`` sample stream; when given,
            the build replays it (used for bit-exact cross-checks).

    Returns:
        The native factorization handle.

    """
    a = sparse.csc_matrix(a, dtype=np.float64)
    a.sort_indices()
    indptr = a.indptr.astype(np.int64)
    indices = a.indices.astype(np.int64)
    data = a.data.astype(np.float64)

    core = native_core()
    if samples is None:
        arrays = core.approx_chol_seeded_f64(indptr, indices, data, int(seed))
    else:
        arrays = core.approx_chol_f64(
            indptr, indices, data, np.ascontiguousarray(samples, dtype=np.float64)
        )
    return NativeLDL(*arrays)
