"""
General batched SDDM solver: one fixed pattern, many matrices and RHS, no loop.

:class:`BatchedSDDMSolver` is built once from a fixed sparsity pattern and routes
*once* (forest -> CHOLMOD) on that pattern.  Each call to :meth:`solve_batch`
hands the chosen native batch engine a single GIL-released call that handles, in
parallel and with no Python loop, either:

* a **shared matrix** with B right-hand sides (``values`` is ``[nnz]``), or
* **B distinct matrices** that share the pattern (``values`` is ``[B, nnz]``),
  each with its own (optionally ragged) block of right-hand sides.

This is the engine the PURC layer assembles values for; it also stands alone for
any fixed-pattern SDDM batch.  Routing mirrors :class:`~purc.laplaciansolve.solver.SDDMSolver`
(forest -> CHOLMOD); per-system approxChol batching is intentionally not provided
(it raises rather than silently looping in Python).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._convert import VectorAdapter, adjacency_to_csc
from ._loader import cholmod_core, has_cholmod, native_core
from .solver import SolverConfig


class BatchedSDDMSolver:
    """Reusable batched solver over a fixed SDDM sparsity pattern."""

    def __init__(self, m, *, config: Optional[SolverConfig] = None) -> None:
        """
        Cache the pattern of *m* and choose the batch backend once.

        Args:
            m: A representative SDDM matrix (scipy / numpy / torch; CPU) whose
                sparsity pattern is shared by every matrix later passed to
                :meth:`solve_batch`.
            config: Solver configuration; ``config.method`` selects the route
                (``"auto"`` / ``"forest"`` / ``"cholmod"``).

        Raises:
            ValueError: If *m* is not square, or ``method="forest"`` on a cyclic
                matrix.
            ImportError: If ``method="cholmod"`` but CHOLMOD was not built.
            NotImplementedError: If neither the forest nor the CHOLMOD route is
                available (per-system approxChol batching is not implemented).

        """
        self.config = config or SolverConfig()
        csc = adjacency_to_csc(m)
        csc.sort_indices()
        if csc.shape[0] != csc.shape[1]:
            raise ValueError(f"matrix must be square, got {csc.shape}")
        self._n = int(csc.shape[0])
        self._nnz = int(csc.nnz)
        self._indptr = csc.indptr.astype(np.int64)
        self._indices = csc.indices.astype(np.int64)
        self._forest = None
        self._cholmod = None
        self.method = self._route(csc)

    def _route(self, csc) -> str:
        """
        Build the batch backend for ``config.method`` and return the route name.

        Args:
            csc: The pattern matrix as a sorted float64 CSC.

        Returns:
            The selected route: ``"forest"`` or ``"cholmod"``.

        Raises:
            ValueError: If ``method="forest"`` but the graph contains a cycle.
            ImportError: If ``method="cholmod"`` but CHOLMOD is unavailable.
            NotImplementedError: If no batched route is available.

        """
        method = self.config.method
        if method in ("auto", "forest"):
            forest = native_core().ForestBatchSolver(self._indptr, self._indices, self._n)
            if forest.is_forest():
                self._forest = forest
                return "forest"
            if method == "forest":
                raise ValueError(
                    "method='forest' requires an acyclic (forest) matrix; cycle detected."
                )
        if method in ("auto", "cholmod"):
            if method == "cholmod" and not has_cholmod():
                raise ImportError(
                    "method='cholmod' requested but the CHOLMOD solver is not built; "
                    "install SuiteSparse and reinstall, or use method='auto'."
                )
            if has_cholmod():
                self._cholmod = cholmod_core().BatchedCholmodSolver(
                    self._indptr, self._indices, csc.data.astype(np.float64), self._n
                )
                return "cholmod"
        raise NotImplementedError(
            "BatchedSDDMSolver supports the forest and CHOLMOD routes; per-system "
            "approxChol batching is not implemented. Install SuiteSparse for the direct "
            "route, or use SDDMSolver for shared-matrix approxChol+PCG."
        )

    @property
    def n(self) -> int:
        """
        The system dimension.

        Returns:
            The number of vertices.

        """
        return self._n

    @property
    def nnz(self) -> int:
        """
        Stored nonzeros of the shared CSC pattern (length of one value block).

        Returns:
            The CSC ``nnz``.

        """
        return self._nnz

    def solve_batch(self, values, rhs, *, rhs_offsets=None):
        """
        Solve a batch of SDDM systems in one GIL-released native call.

        Args:
            values: Matrix values in the cached CSC order - ``[nnz]`` for a single
                shared matrix, or ``[B, nnz]`` for B matrices sharing the pattern.
            rhs: A ``[B, n]`` array (one RHS per system, or B RHS for a shared
                matrix) or, with ``rhs_offsets``, a flat ``[totalK, n]`` block.
            rhs_offsets: Optional int64 ``[B+1]`` offsets so system ``s`` owns rhs
                rows ``[rhs_offsets[s], rhs_offsets[s+1])`` (ragged RHS per system).

        Returns:
            The solutions in the same array kind / dtype / device as ``rhs``,
            shaped like ``rhs``.

        Raises:
            ValueError: On shape / pattern mismatches.
            numpy.linalg.LinAlgError: If a CHOLMOD system is not positive definite.

        """
        vals = np.ascontiguousarray(VectorAdapter(values).array, dtype=np.float64)
        shared = vals.ndim == 1
        if not shared and vals.ndim != 2:
            raise ValueError("values must be 1-D [nnz] (shared) or 2-D [B, nnz] (per-system)")
        if vals.shape[-1] != self._nnz:
            raise ValueError(f"values last dim {vals.shape[-1]} != pattern nnz {self._nnz}")

        adapter = VectorAdapter(rhs)
        r = adapter.array
        if r.ndim != 2:
            raise ValueError("rhs must be a 2-D [B, n] (or [totalK, n] with rhs_offsets) array")
        r = np.ascontiguousarray(r, dtype=np.float64)
        if r.shape[1] != self._n:
            raise ValueError(f"rhs has {r.shape[1]} columns, expected n={self._n}")
        total = r.shape[0]
        out = np.empty((total, self._n), dtype=np.float64)

        if shared:
            self._solve_shared(vals, r, out)
        else:
            batch = vals.shape[0]
            if rhs_offsets is None:
                if total != batch:
                    raise ValueError(
                        f"per-system rhs must have B={batch} rows when rhs_offsets is None, "
                        f"got {total}"
                    )
                offs = np.arange(batch + 1, dtype=np.int64)
            else:
                offs = np.ascontiguousarray(rhs_offsets, dtype=np.int64)
                if offs.shape[0] != batch + 1 or int(offs[-1]) != total:
                    raise ValueError(
                        "rhs_offsets must be a [B+1] array with rhs_offsets[-1] == number of "
                        "RHS rows"
                    )
            self._solve_per_system(vals, batch, r, offs, out)
        return adapter.restore(out)

    def _solve_shared(self, values: np.ndarray, rhs: np.ndarray, out: np.ndarray) -> None:
        """
        Solve one shared matrix against all rows of *rhs*.

        Args:
            values: The single matrix's values ``[nnz]``.
            rhs: Right-hand sides ``[B, n]``.
            out: Output buffer ``[B, n]`` (filled in place).

        """
        if self.method == "forest":
            self._forest.solve_batch_shared(values, rhs, out)
            return
        # CHOLMOD: one factorization (B=1), all RHS as a single block (ncol=B).
        vals2d = np.ascontiguousarray(values[None, :])
        offs = np.array([0, rhs.shape[0]], dtype=np.int64)
        status = np.empty(1, dtype=np.int64)
        self._cholmod.factorize_and_solve_batch(vals2d, rhs, offs, out, status)
        self._check_status(status)

    def _solve_per_system(
        self, values: np.ndarray, batch: int, rhs: np.ndarray, offs: np.ndarray, out: np.ndarray
    ) -> None:
        """
        Solve B distinct matrices, each against its own RHS block.

        Args:
            values: Per-system values ``[B, nnz]``.
            batch: The number of systems B.
            rhs: Flat right-hand sides ``[totalK, n]``.
            offs: Int64 ``[B+1]`` RHS-block offsets.
            out: Output buffer ``[totalK, n]`` (filled in place).

        """
        if self.method == "forest":
            self._forest.solve_batch(values, rhs, offs, out)
            return
        status = np.empty(batch, dtype=np.int64)
        self._cholmod.factorize_and_solve_batch(values, rhs, offs, out, status)
        self._check_status(status)

    @staticmethod
    def _check_status(status: np.ndarray) -> None:
        """
        Raise if any CHOLMOD per-system factorization failed.

        Args:
            status: Per-system status codes (0 = ok).

        Raises:
            numpy.linalg.LinAlgError: If any system was not positive definite.

        """
        bad = np.flatnonzero(status != 0)
        if bad.size:
            raise np.linalg.LinAlgError(
                f"CHOLMOD batch factorization failed for system(s) {bad.tolist()} "
                "(matrix not positive definite?)."
            )
