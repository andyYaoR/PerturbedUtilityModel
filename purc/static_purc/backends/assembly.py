r"""
Fixed-pattern assembly of the Newton matrix ``M = A diag(w) A^T + eps I``.

Across SSN iterations only the per-coordinate weights ``w = D * active_mask`` and
the regularizer ``eps`` change; the sparsity pattern of ``M`` is fixed (it is the
pattern of ``A A^T`` together with the diagonal).  :class:`CSCAssembler`
precomputes, once, the "triple-product" scatter map -- for every structural
nonzero ``(r, c)`` of ``M`` the list of coordinates ``i`` with ``A[r,i] A[c,i] !=
0`` and their products -- so each assembly is a single vectorized scatter-add
into a CSC value array of fixed length.  This is the reference implementation of
the kernel ported to C++ in v0.3.0.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..utils.native import native_available, native_core


class CSCAssembler:
    """
    Precomputed assembler for ``M = A diag(w) A^T + eps I`` at a fixed pattern.

    Args:
        A: The ``(k, N)`` constraint matrix (any scipy sparse / dense; coerced to
            CSC internally).

    Attributes:
        shape: The ``(k, k)`` shape of ``M``.
        nnz: Number of stored nonzeros in the fixed pattern.

    """

    def __init__(self, A) -> None:
        Acsc = sp.csc_matrix(A).astype(float)
        k = Acsc.shape[0]
        indptr, indices, values = Acsc.indptr, Acsc.indices, Acsc.data

        rows: list[int] = []
        cols: list[int] = []
        coeff: list[float] = []
        srci: list[int] = []
        # Each column i (a coordinate) contributes the outer product of its
        # nonzero rows: M[r, c] += A[r,i] A[c,i] w_i for all r, c in support(i).
        for i in range(Acsc.shape[1]):
            seg = slice(indptr[i], indptr[i + 1])
            ri = indices[seg]
            vi = values[seg]
            for a in range(ri.size):
                for b in range(ri.size):
                    rows.append(int(ri[a]))
                    cols.append(int(ri[b]))
                    coeff.append(float(vi[a] * vi[b]))
                    srci.append(i)
        # Ensure every diagonal entry exists in the pattern so eps*I always lands,
        # even for a row whose Laplacian diagonal happens to vanish.
        for j in range(k):
            rows.append(j)
            cols.append(j)
            coeff.append(0.0)
            srci.append(0)

        rows_a = np.asarray(rows, dtype=np.int64)
        cols_a = np.asarray(cols, dtype=np.int64)
        self._coeff = np.asarray(coeff, dtype=float)
        self._srci = np.asarray(srci, dtype=np.int64)

        # Canonical CSC pattern (deduplicated, sorted) defines the fixed layout.
        canonical = sp.coo_matrix((np.zeros(rows_a.size), (rows_a, cols_a)), shape=(k, k)).tocsc()
        canonical.sum_duplicates()
        self.indptr = canonical.indptr
        self.indices = canonical.indices
        self.shape = (k, k)
        self.nnz = int(canonical.nnz)

        # Map each contribution and each diagonal to its slot in canonical .data.
        slot_of: dict[tuple[int, int], int] = {}
        for col in range(k):
            for p in range(canonical.indptr[col], canonical.indptr[col + 1]):
                slot_of[(int(canonical.indices[p]), col)] = p
        self._slot = np.asarray(
            [slot_of[(int(r), int(c))] for r, c in zip(rows_a, cols_a)], dtype=np.int64
        )
        self._diag_slot = np.asarray([slot_of[(j, j)] for j in range(k)], dtype=np.int64)
        # Scatter map ``[nnz, M]`` so a whole batch assembles as one sparse matmul:
        # ``values[B, nnz] = (Smap @ contrib[B, M].T).T`` with ``contrib = coeff * w[srci]``.
        m = self._coeff.size
        self._scatter = sp.csr_matrix((np.ones(m), (self._slot, np.arange(m))), shape=(self.nnz, m))

    def assemble_values(self, w: np.ndarray, eps: float) -> np.ndarray:
        """
        Compute the CSC value array of ``M = A diag(w) A^T + eps I``.

        Args:
            w: Per-coordinate weights ``D * active_mask``, shape ``(N,)``.
            eps: Diagonal regularizer added to every diagonal entry.

        Returns:
            The ``(nnz,)`` value array aligned to :attr:`indices` / :attr:`indptr`.

        """
        w = np.ascontiguousarray(w, dtype=float)
        if native_available():
            out = np.empty(self.nnz, dtype=float)
            native_core().csc_assemble_f64(
                self._slot, self._coeff, self._srci, w, self._diag_slot, float(eps), out
            )
            return out
        # Fallback: bincount is the fast scatter-add (np.add.at is markedly slower);
        # slots are precomputed in [0, nnz), so minlength pins the output length.
        data = np.bincount(self._slot, weights=self._coeff * w[self._srci], minlength=self.nnz)
        data[self._diag_slot] += eps
        return data

    def assemble_values_batch(self, w: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """
        Assemble the CSC value arrays for a whole batch in one vectorized scatter.

        For ``B`` systems sharing the pattern, computes ``values[B, nnz]`` via a
        single sparse matmul (``self._scatter @ contrib.T``) rather than ``B``
        per-system scatters -- the fast path for the batched general solve.

        Args:
            w: Per-system weights ``[B, N]``.
            eps: Per-system diagonal regularizers ``[B]``.

        Returns:
            The ``[B, nnz]`` value arrays aligned to :attr:`indices` / :attr:`indptr`.

        """
        w = np.ascontiguousarray(w, dtype=float)
        contrib = self._coeff[None, :] * w[:, self._srci]  # [B, M]
        out = (self._scatter @ contrib.T).T  # [B, nnz]
        out = np.ascontiguousarray(out)
        out[:, self._diag_slot] += np.asarray(eps, dtype=float)[:, None]
        return out

    def assemble(self, w: np.ndarray, eps: float) -> sp.csc_matrix:
        """
        Assemble ``M`` as a CSC matrix at the fixed pattern.

        Args:
            w: Per-coordinate weights ``D * active_mask``, shape ``(N,)``.
            eps: Diagonal regularizer.

        Returns:
            ``M`` as a ``scipy.sparse.csc_matrix`` with the cached pattern.

        """
        data = self.assemble_values(w, eps)
        return sp.csc_matrix((data, self.indices, self.indptr), shape=self.shape)
