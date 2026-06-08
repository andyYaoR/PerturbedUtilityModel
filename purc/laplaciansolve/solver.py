"""
Reusable, auto-caching Laplacian / SDDM solver handles.

This is the high-level API tuned for the PURC semismooth-Newton loop, where the
graph *sparsity pattern is fixed* across all iterations and OD pairs while only
the edge weights, the RHS, and the regularizer change.  A handle is built once
and then:

* **reuses** the approxChol factorization as a PCG preconditioner across solves
  (the build - the expensive sequential step - is paid once, not per solve);
* supports **re-solve without rebuild** via :meth:`LaplacianSolver.update_weights`
  (refill the matrix values on the cached pattern; the stale preconditioner is
  reused and only rebuilt when PCG iterations degrade past a threshold);
* **warm-starts** PCG from a supplied iterate;
* solves a **batch** of right-hand sides via a single GIL-released native call
  that parallelizes over the systems with OpenMP (no Python loop, no GIL);
* takes the exact **forest fast path** (one apply, no PCG) when the graph is
  acyclic.

The native build/solve kernels do the heavy lifting; graph preprocessing
(Laplacian assembly, components, forest detection) reuses the shared utilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy import sparse

from ._convert import VectorAdapter, adjacency_to_csc
from ._loader import native_core
from .factorization import NativeLDL, build_approxchol
from .reference.forest_ldl import is_forest
from .reference.graph import lap, validate_adjacency
from .reference.sddm import adj_val_and_excess, extend_matrix

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class SolverConfig:
    """
    Configuration for a reusable solver handle.

    Attributes:
        tol: PCG relative-residual tolerance.
        maxits: PCG iteration cap.
        stag_test: PCG stagnation window (``0`` disables).
        seed: Seed for the approxChol build RNG.
        rebuild_factor: Rebuild the preconditioner once a solve's iteration count
            exceeds ``rebuild_factor`` times the count right after a (re)build.
        max_workers: Thread-pool size for batch solves (``None`` = os.cpu_count).
        method: SDDM route selection - ``"auto"`` (forest LDL if acyclic, else
            direct CHOLMOD when available, else approxChol+PCG), ``"forest"``
            (force exact forest LDL; error if cyclic), ``"cholmod"`` (force
            direct; error if unavailable), or ``"approxchol"`` (force iterative).

    """

    tol: float = 1e-6
    maxits: int = 1000
    stag_test: int = 5
    seed: int = 0
    rebuild_factor: float = 2.0
    max_workers: Optional[int] = None
    method: str = "auto"

    def __post_init__(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If ``tol``/``maxits``/``rebuild_factor``/``method`` are invalid.

        """
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol}")
        if self.maxits < 1:
            raise ValueError(f"maxits must be >= 1, got {self.maxits}")
        if self.rebuild_factor < 1.0:
            raise ValueError(f"rebuild_factor must be >= 1, got {self.rebuild_factor}")
        if self.method not in ("auto", "forest", "cholmod", "approxchol"):
            raise ValueError(
                f"method must be 'auto'/'forest'/'cholmod'/'approxchol', got {self.method!r}"
            )


def _lap_csc_int64(a: sparse.csc_matrix):
    """
    Return ``lap(a)`` as a sorted CSC with int64 index arrays.

    Args:
        a: Adjacency matrix.

    Returns:
        ``(indptr, indices, data)`` for the native PCG matvec.

    """
    la = lap(a).tocsc()
    la.sort_indices()
    return la.indptr.astype(np.int64), la.indices.astype(np.int64), la.data.astype(np.float64)


class LaplacianSolver:
    """
    A reusable handle solving Laplacian systems of a fixed-pattern adjacency.

    The approxChol preconditioner is built once at construction and reused across
    :meth:`solve` / :meth:`solve_batch` / :meth:`update_weights` calls; it is
    rebuilt only when PCG iterations degrade past ``config.rebuild_factor``.
    """

    def __init__(self, a, *, config: Optional[SolverConfig] = None) -> None:
        """
        Build the handle and its (reused) preconditioner.

        Args:
            a: Symmetric, nonnegative, zero-diagonal adjacency matrix.
            config: Solver configuration; defaults to :class:`SolverConfig`.

        """
        self.config = config or SolverConfig()
        self._a = adjacency_to_csc(a)
        validate_adjacency(self._a)
        self._n = self._a.shape[0]
        self._forest = is_forest(self._a)
        self._indptr, self._indices, self._data = _lap_csc_int64(self._a)
        self._ldli: NativeLDL = build_approxchol(self._a, seed=self.config.seed)
        self._baseline_iters = 0  # set on first solve (right after a build)
        self._last_iters = 0

    @property
    def n(self) -> int:
        """
        Number of vertices.

        Returns:
            The system dimension.

        """
        return self._n

    @property
    def is_forest(self) -> bool:
        """
        Whether the graph is acyclic (uses the exact, no-PCG fast path).

        Returns:
            ``True`` if the cached graph is a forest.

        """
        return self._forest

    def rebuild_preconditioner(self) -> None:
        """Rebuild the approxChol preconditioner from the current weights."""
        self._ldli = build_approxchol(self._a, seed=self.config.seed)
        self._baseline_iters = 0

    def update_weights(self, a) -> None:
        """
        Refill the matrix values on the cached pattern (re-solve without rebuild).

        The new adjacency must share the cached sparsity pattern; only the edge
        weights change.  The existing preconditioner is kept (reused as a stale
        but still-valid PCG preconditioner) and rebuilt lazily by the iteration
        policy in :meth:`solve`.

        Args:
            a: New adjacency with the same pattern as the original.

        Raises:
            ValueError: If the new adjacency's pattern differs from the cached one.

        """
        new = adjacency_to_csc(a)
        if (
            new.nnz != self._a.nnz
            or not np.array_equal(new.indptr, self._a.indptr)
            or not np.array_equal(new.indices, self._a.indices)
        ):
            raise ValueError("update_weights requires the same sparsity pattern")
        self._a = new
        self._indptr, self._indices, self._data = _lap_csc_int64(new)

    def _maybe_rebuild(self) -> None:
        """Rebuild the preconditioner if the last solve's iterations degraded."""
        if self._baseline_iters and self._last_iters > self.config.rebuild_factor * max(
            self._baseline_iters, 1
        ):
            self.rebuild_preconditioner()

    def _solve_once(self, b: np.ndarray, x0: Optional[np.ndarray]) -> tuple[np.ndarray, int, bool]:
        """
        Pure single-RHS solve against the current matrix + cached preconditioner.

        Used by :meth:`solve`; the batch path uses the native batched kernel
        directly rather than looping this in Python.

        Args:
            b: Right-hand side of length ``n``.
            x0: Optional warm-start iterate.

        Returns:
            ``(solution, pcg_iterations, converged)``; the exact forest path
            returns ``(x, 0, True)``.

        """
        bz = b - b.mean()
        ldli = self._ldli
        if self._forest:
            return ldli.solve(bz, subtract_mean=True), 0, True
        x = np.zeros(self._n, dtype=np.float64) if x0 is None else np.array(x0, dtype=np.float64)
        its, _relres, conv = native_core().pcg_lap_f64(
            self._indptr,
            self._indices,
            self._data,
            bz,
            ldli.col,
            ldli.colptr,
            ldli.rowval,
            ldli.fval,
            ldli.d,
            x,
            self.config.tol,
            self.config.maxits,
            self.config.stag_test,
        )
        return x, its, bool(conv)

    def _record_and_maybe_rebuild(self, iters: int) -> None:
        """
        Update the iteration counters and proactively rebuild if iterations climb.

        Args:
            iters: The (max) PCG iteration count of the just-completed solve(s).

        """
        self._last_iters = iters
        if self._baseline_iters == 0 and iters > 0:
            self._baseline_iters = iters
        self._maybe_rebuild()

    def solve(self, b: ArrayLike, *, x0: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Solve ``lap(a) x = b - mean(b)`` reusing the cached preconditioner.

        Reuses the cached preconditioner; if it has grown too stale for PCG to
        converge, the preconditioner is rebuilt from the current matrix and the
        solve is retried (guaranteeing a converged result, never a silent
        fallback to a wrong answer).

        Args:
            b: Right-hand side of length ``n``.
            x0: Optional warm-start iterate.

        Returns:
            The solution vector.

        """
        adapter = VectorAdapter(b)
        b_np = adapter.array
        start = None if x0 is None else VectorAdapter(x0).array
        x, its, conv = self._solve_once(b_np, start)
        if not conv and not self._forest:
            self.rebuild_preconditioner()  # stale -> match the current matrix and retry
            x, its, conv = self._solve_once(b_np, start)
        self._record_and_maybe_rebuild(its)
        return adapter.restore(x)

    def solve_batch(
        self,
        rhs: Union[np.ndarray, List[ArrayLike]],
        *,
        x0: Optional[Union[np.ndarray, List[ArrayLike]]] = None,
    ) -> np.ndarray:
        """
        Solve a batch of right-hand sides natively in parallel.

        All right-hand sides share the handle's current matrix and cached
        preconditioner.  The entire batch is one GIL-released native call that
        parallelizes internally with OpenMP over the systems (no Python loop, no
        Python thread pool).  If the shared preconditioner is too stale for some
        systems to converge, it is rebuilt once and those systems are re-solved
        (again natively).

        Args:
            rhs: Either a ``(B, n)`` array or a list of ``B`` length-``n`` vectors.
            x0: Optional warm starts, same shape as ``rhs``.

        Returns:
            A ``(B, n)`` array of solutions.

        """
        adapter = VectorAdapter(rhs)
        rhs_arr = adapter.array
        if rhs_arr.ndim == 1:
            rhs_arr = rhs_arr[None, :]
        batch = rhs_arr.shape[0]
        # Mean-center every row (vectorized; the Laplacian RHS must be mean-zero).
        bz = np.ascontiguousarray(rhs_arr - rhs_arr.mean(axis=1, keepdims=True))
        core = native_core()
        ldli = self._ldli

        if self._forest:
            out = np.zeros((batch, self._n), dtype=np.float64)
            core.ldl_solve_batch_f64(
                ldli.col, ldli.colptr, ldli.rowval, ldli.fval, ldli.d, bz, out, True
            )
            return adapter.restore(out)

        out = np.zeros((batch, self._n), dtype=np.float64)
        if x0 is not None:
            x0_arr = VectorAdapter(x0).array
            out[:] = x0_arr if x0_arr.ndim == 2 else x0_arr[None, :]
        iters = np.zeros(batch, dtype=np.int64)
        conv = np.zeros(batch, dtype=np.int64)

        workers = self.config.max_workers or (os.cpu_count() or 1)
        orig = core.max_threads()
        core.set_threads(max(1, min(workers, batch)))  # thread pool over the batch
        try:
            core.pcg_lap_batch_f64(
                self._indptr,
                self._indices,
                self._data,
                bz,
                ldli.col,
                ldli.colptr,
                ldli.rowval,
                ldli.fval,
                ldli.d,
                out,
                iters,
                conv,
                self.config.tol,
                self.config.maxits,
                self.config.stag_test,
            )
            if not conv.all():
                # One rebuild fixes the shared matrix; re-solve the stale rows.
                self.rebuild_preconditioner()
                ldli = self._ldli
                stale = np.flatnonzero(conv == 0)
                sub_b = np.ascontiguousarray(bz[stale])
                sub_x = np.ascontiguousarray(out[stale])
                si = np.zeros(stale.size, dtype=np.int64)
                sc = np.zeros(stale.size, dtype=np.int64)
                core.pcg_lap_batch_f64(
                    self._indptr,
                    self._indices,
                    self._data,
                    sub_b,
                    ldli.col,
                    ldli.colptr,
                    ldli.rowval,
                    ldli.fval,
                    ldli.d,
                    sub_x,
                    si,
                    sc,
                    self.config.tol,
                    self.config.maxits,
                    self.config.stag_test,
                )
                out[stale] = sub_x
                iters[stale] = si
        finally:
            core.set_threads(orig)

        self._record_and_maybe_rebuild(int(iters.max()) if batch else 0)
        return adapter.restore(out)


class _GroundedApproxCholSDDM:
    """
    Iterative SDDM backend: grounding embedding + cached approxChol PCG.

    ``M`` is embedded into the Laplacian of an augmented graph (a ground vertex
    carries each row's diagonal-dominance excess) and solved with a cached
    :class:`LaplacianSolver` on that augmented graph; the ground entry is then
    dropped.  Built once and reused across right-hand sides and weight updates,
    with native batched solves.  Used by :class:`SDDMSolver` when CHOLMOD is
    unavailable or ``method="approxchol"``.
    """

    def __init__(self, m, *, config: Optional[SolverConfig] = None) -> None:
        """
        Build the grounded Laplacian solver.

        Args:
            m: A symmetric SDDM matrix (scipy / numpy / torch; CPU).
            config: Solver configuration; defaults to :class:`SolverConfig`.

        """
        self.config = config or SolverConfig()
        self._build(m)

    def _build(self, m) -> None:
        """
        Construct the grounded augmented Laplacian handle from *m*.

        Args:
            m: The SDDM matrix.

        """
        a, _diag, excess = adj_val_and_excess(adjacency_to_csc(m))
        a1, extended = extend_matrix(a, excess)
        self._n = a.shape[0]
        self._extended = extended
        self._lap = LaplacianSolver(a1, config=self.config)

    @property
    def n(self) -> int:
        """
        Dimension of the SDDM system.

        Returns:
            The number of original (non-ground) vertices.

        """
        return self._n

    def solve(self, b, *, x0=None):
        """
        Solve ``M x = b`` for a single right-hand side.

        Args:
            b: Right-hand side of length ``n`` (numpy / scipy / torch).
            x0: Optional warm start (ground entry appended internally).

        Returns:
            The solution in the same array kind / dtype / device as ``b``.

        """
        adapter = VectorAdapter(b)
        b_np = adapter.array
        if not self._extended:  # already a Laplacian (zero excess)
            return adapter.restore(self._lap.solve(b_np, x0=x0))
        b_aug = np.concatenate([b_np, [-float(b_np.sum())]])
        start = None if x0 is None else np.concatenate([VectorAdapter(x0).array, [0.0]])
        x_aug = self._lap.solve(b_aug, x0=start)
        x_aug = x_aug - x_aug[-1]
        return adapter.restore(x_aug[: self._n])

    def solve_batch(self, rhs, *, x0=None):
        """
        Solve ``M x = b`` for a batch of right-hand sides (native, parallel).

        Args:
            rhs: A ``(B, n)`` array or list of length-``n`` vectors
                (numpy / scipy / torch).
            x0: Optional warm starts, same shape as ``rhs``.

        Returns:
            A ``(B, n)`` solution in the same array kind as ``rhs``.

        """
        adapter = VectorAdapter(rhs)
        r = adapter.array
        if r.ndim == 1:
            r = r[None, :]
        if not self._extended:
            return adapter.restore(self._lap.solve_batch(r, x0=x0))
        ground = -r.sum(axis=1, keepdims=True)
        r_aug = np.ascontiguousarray(np.concatenate([r, ground], axis=1))
        start = None
        if x0 is not None:
            xs = VectorAdapter(x0).array
            if xs.ndim == 1:
                xs = xs[None, :]
            start = np.ascontiguousarray(np.concatenate([xs, np.zeros((xs.shape[0], 1))], axis=1))
        x_aug = self._lap.solve_batch(r_aug, x0=start)
        x_aug = x_aug - x_aug[:, -1:]
        return adapter.restore(x_aug[:, : self._n])

    def update(self, m) -> None:
        """
        Re-point the handle at an updated SDDM matrix without rebuilding.

        Reuses the cached preconditioner when the (augmented) sparsity pattern is
        unchanged (only weights / regularizer differ); otherwise rebuilds.

        Args:
            m: The updated SDDM matrix (same graph pattern for the fast path).

        """
        a, _diag, excess = adj_val_and_excess(adjacency_to_csc(m))
        a1, extended = extend_matrix(a, excess)
        if extended == self._extended:
            try:
                self._lap.update_weights(a1)
                return
            except ValueError:
                pass
        self._n = a.shape[0]
        self._extended = extended
        self._lap = LaplacianSolver(a1, config=self.config)


def _forest_gate(m) -> bool:
    """
    O(1) necessary check that *m*'s off-diagonal graph could be a forest.

    A forest on ``n`` vertices has at most ``n-1`` edges, so a (symmetric) matrix
    with ``>= n`` off-diagonal edges is certainly cyclic.  Cheap for scipy sparse
    input (uses ``nnz``); for dense / torch input it returns ``True`` and lets the
    native leaf-pruning make the exact decision.

    Args:
        m: Candidate SDDM matrix.

    Returns:
        ``True`` if a forest is still possible (worth attempting detection).

    """
    if sparse.issparse(m):
        n = m.shape[0]
        n_edges = max(0, (m.nnz - n) // 2)  # assume a full diagonal (SDDM has eps*I)
        return n_edges <= n - 1
    return True


class SDDMSolver:
    """
    Reusable solver for SDDM systems ``M x = b`` (e.g. the PURC ``H + eps*I``).

    Auto-selects the optimal route (the "choose the best method" subroutine), in
    order of preference:

    1. **Forest LDL** - if the off-diagonal graph is acyclic (the converged PURC
       regime), an exact zero-fill O(n) factorization; detected cheaply via an
       O(1) edge-count gate plus a native leaf-pruning confirm.
    2. **Direct CHOLMOD** - a sparse Cholesky for general SPD / low-fill systems
       (road networks), matching/beating Julia's ``cholesky``.
    3. **Grounding + approxChol PCG** - the iterative path for large / high-fill
       graphs.

    Force a route with ``SolverConfig(method="forest" | "cholmod" | "approxchol")``;
    ``"forest"`` raises if the matrix is not acyclic. Built once and reused across
    right-hand sides and weight/regularizer updates; supports native batched solves.
    """

    def __init__(self, m, *, config: Optional[SolverConfig] = None) -> None:
        """
        Build the SDDM handle, choosing the backend per ``config.method``.

        Args:
            m: A symmetric SDDM matrix (scipy / numpy / torch; CPU).
            config: Solver configuration; defaults to :class:`SolverConfig`.

        Raises:
            ImportError: If ``method="cholmod"`` but CHOLMOD was not built.
            ValueError: If ``method="forest"`` but the graph contains a cycle.

        """
        from ._loader import has_cholmod

        self.config = config or SolverConfig()
        method = self.config.method

        # 1) Forest fast path: exact, zero-fill, O(n).  "auto" tries it behind the
        #    cheap O(1) gate; "forest" forces it (and errors on a cyclic graph).
        if method in ("auto", "forest") and (method == "forest" or _forest_gate(m)):
            from .forest_solver import ForestLDLSDDM

            backend = ForestLDLSDDM(m, config=self.config)
            if backend.is_forest:
                self._backend = backend
                self.method = "forest"
                return
            if method == "forest":
                raise ValueError(
                    "method='forest' requires an acyclic (forest) matrix; cycle detected."
                )

        # 2) Direct CHOLMOD for general SPD / low-fill systems.
        use_cholmod = method == "cholmod" or (method == "auto" and has_cholmod())
        if method == "cholmod" and not has_cholmod():
            raise ImportError(
                "method='cholmod' requested but the CHOLMOD solver is not built; "
                "install SuiteSparse and reinstall, or use method='auto'/'approxchol'."
            )
        if use_cholmod:
            from .cholmod_solver import CholmodSDDM

            self._backend = CholmodSDDM(m, config=self.config)
            self.method = "cholmod"
            return

        # 3) Grounding + approxChol PCG for large / high-fill graphs.
        self._backend = _GroundedApproxCholSDDM(m, config=self.config)
        self.method = "approxchol"

    @property
    def n(self) -> int:
        """
        Dimension of the SDDM system.

        Returns:
            The number of vertices.

        """
        return self._backend.n

    def solve(self, b, *, x0=None):
        """
        Solve ``M x = b`` for one right-hand side (delegated to the backend).

        Args:
            b: Right-hand side (numpy / scipy / torch).
            x0: Optional warm start (used only by the iterative backend).

        Returns:
            The solution in the same array kind as ``b``.

        """
        return self._backend.solve(b, x0=x0)

    def solve_batch(self, rhs, *, x0=None):
        """
        Solve ``M x = b`` for a batch of right-hand sides (native, parallel).

        Args:
            rhs: A ``(B, n)`` array or list of length-``n`` vectors.
            x0: Optional warm starts (iterative backend only).

        Returns:
            A ``(B, n)`` solution in the same array kind as ``rhs``.

        """
        return self._backend.solve_batch(rhs, x0=x0)

    def update(self, m) -> None:
        """
        Re-solve an updated SDDM matrix without rebuilding the symbolic factor.

        Args:
            m: The updated SDDM matrix (same sparsity pattern for the fast path).

        """
        self._backend.update(m)
