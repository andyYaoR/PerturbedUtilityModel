"""
The ``LDLinv`` factorization container and its forward/backward solve.

Faithful port of ``LDLinv`` and ``LDLsolver`` / ``forward!`` / ``backward!``
from ``Laplacians.jl`` (``src/approxChol.jl``), in 0-based indexing.

``LDLinv`` represents an approximate factorization ``L D L^T`` of the Laplacian.
A solve applies ``L^{-1}``, then ``D^{-1}``, then ``L^{-T}`` via two triangular
sweeps over the elimination order, and finally projects out the constant
(null-space) component for the Laplacian case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LDLinv:
    """
    Approximate ``L D L^T`` factorization produced by :func:`approx_chol`.

    Attributes:
        col: Elimination order; ``col[ii]`` is the vertex eliminated ``ii``-th.
            Length ``n - 1`` (the last vertex is never eliminated).
        colptr: CSR-style pointers; the entries of the ``ii``-th eliminated
            column occupy ``rowval[colptr[ii]:colptr[ii + 1]]``.  Length ``n``.
        rowval: Row indices of the off-diagonal ``L`` entries.
        fval: Elimination multipliers (the ``f = w / wdeg`` fractions); the last
            entry of each column is ``1.0``.
        d: Diagonal of ``D``; ``d[i] == 0`` marks a vertex with no pivot.

    """

    col: np.ndarray
    colptr: np.ndarray
    rowval: np.ndarray
    fval: np.ndarray
    d: np.ndarray

    @property
    def n(self) -> int:
        """
        Number of vertices.

        Returns:
            The dimension ``n`` of the factorized matrix.

        """
        return int(self.d.shape[0])

    def nnz(self) -> int:
        """
        Number of stored ``L`` entries (operator size proxy).

        Returns:
            ``len(fval)``.

        """
        return int(self.fval.shape[0])


def _forward(ldli: LDLinv, y: np.ndarray) -> None:
    """
    Apply ``L^{-1}`` in place (forward substitution over the elim order).

    Args:
        ldli: The factorization.
        y: Working vector, modified in place.

    """
    col = ldli.col
    colptr = ldli.colptr
    rowval = ldli.rowval
    fval = ldli.fval
    for ii in range(col.shape[0]):
        i = col[ii]
        j0 = colptr[ii]
        j1 = colptr[ii + 1] - 1
        yi = y[i]
        for jj in range(j0, j1):
            j = rowval[jj]
            f = fval[jj]
            y[j] += f * yi
            yi *= 1.0 - f
        j = rowval[j1]
        y[j] += yi
        y[i] = yi


def _backward(ldli: LDLinv, y: np.ndarray) -> None:
    """
    Apply ``L^{-T}`` in place (backward substitution over the elim order).

    Args:
        ldli: The factorization.
        y: Working vector, modified in place.

    """
    col = ldli.col
    colptr = ldli.colptr
    rowval = ldli.rowval
    fval = ldli.fval
    for ii in range(col.shape[0] - 1, -1, -1):
        i = col[ii]
        j0 = colptr[ii]
        j1 = colptr[ii + 1] - 1
        j = rowval[j1]
        yi = y[i] + y[j]
        for jj in range(j1 - 1, j0 - 1, -1):
            j = rowval[jj]
            f = fval[jj]
            yi = (1.0 - f) * yi + f * y[j]
        y[i] = yi


def ldl_solve(ldli: LDLinv, b: np.ndarray, *, subtract_mean: bool = True) -> np.ndarray:
    """
    Solve ``(L D L^T) x = b`` for a single right-hand side.

    Args:
        ldli: The approximate factorization.
        b: Right-hand side vector of length ``n``.
        subtract_mean: If ``True`` (default, matching Julia), project the result
            onto the mean-zero subspace — required for the singular Laplacian.
            Set ``False`` only for nonsingular (SDDM) systems where the constant
            null space is absent, to save one pass.

    Returns:
        The solution vector ``x`` of length ``n``.

    """
    y = np.array(b, dtype=np.float64, copy=True)
    _forward(ldli, y)

    d = ldli.d
    nonzero = d != 0.0
    y[nonzero] /= d[nonzero]

    _backward(ldli, y)

    if subtract_mean:
        y -= y.mean()
    return y
