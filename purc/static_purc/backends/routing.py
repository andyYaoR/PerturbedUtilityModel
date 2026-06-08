r"""
Backend dispatcher for the per-Newton-iteration linear solve.

The Newton system ``(A diag(w) A^T + eps I) d = rhs`` has a fixed sparsity
pattern, so the LaplacianSolve handle is built once and reused across iterations
-- only ``w``, ``eps`` and ``rhs`` change.  :class:`LaplacianBackend` builds the
right LaplacianSolve solver for the constraint geometry:

* **simple incidence A**: route to ``PURCLaplacianSolver`` for the network fast
  path (fused native assembly + solve, forest route when acyclic).  Parallel
  arcs (two-way roads) are not supported by it and fall back to the general path.
* **general A**: assemble ``M`` via :class:`CSCAssembler` and drive
  ``SDDMSolver``.  Because a general ``A diag(w) A^T`` is SPD but *not* necessarily
  an M-matrix (off-diagonals may be positive), we force ``method="cholmod"`` (a
  true general SPD Cholesky) when CHOLMOD is available.

Both single (`solve`) and batched (`solve_batch`, one system per OD-pair) entry
points are provided.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from ..constraints.base import Polytope
from ..utils.logging import get_logger
from ..utils.torch_compat import as_tensor, to_numpy
from .assembly import CSCAssembler

_logger = get_logger(__name__)


class LaplacianBackend:
    """
    Build-once / solve-many wrapper around a LaplacianSolve SDDM solver.

    Args:
        polytope: The constraint polytope (its ``A`` fixes the pattern).
        laplacian_options: Options forwarded to ``purc.laplaciansolve.SolverConfig``
            (e.g. ``{"method": "cholmod", "tol": 1e-10}``).  ``method`` defaults
            to ``"cholmod"`` when available, else ``"auto"``.

    """

    def __init__(
        self,
        polytope: Polytope,
        laplacian_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        opts = dict(laplacian_options or {})
        self.k = polytope.num_constraints
        self._solver = None  # general path: built lazily on first solve
        self._batched = None  # general path: BatchedSDDMSolver, built lazily on first batch
        self._prep: Optional[tuple] = None  # PURC path: cached (w, eps) for solve_prepared

        purc = None
        if getattr(polytope, "is_incidence", False) and polytope.edges is not None:
            # Network fast path: hand the edge list to PURCLaplacianSolver, which
            # fuses assembly + solve in one GIL-released native call and routes to
            # forest (acyclic) or CHOLMOD.  Let it auto-route (do not force cholmod).
            # PURCLaplacianSolver rejects parallel arcs (two-way roads); for such
            # multigraph incidence we fall back to the general SDDM path, which
            # handles it correctly (the assembler sums parallel-edge contributions).
            try:
                from purc.laplaciansolve import PURCLaplacianSolver, SolverConfig

                purc = PURCLaplacianSolver(
                    polytope.edges, n_nodes=polytope.n_nodes, config=SolverConfig(**opts)
                )
            except Exception as exc:  # pragma: no cover - exercised by multigraphs
                _logger.debug("PURCLaplacianSolver unavailable (%s); using SDDM path", exc)
                purc = None

        if purc is not None:
            self.kind = "purc"
            self._purc = purc
            self.method = purc.method
            self.phase = purc.phase
            _logger.debug("incidence detected: using PURCLaplacianSolver (%s)", self.method)
        else:
            # General path: assemble M and drive SDDMSolver (CHOLMOD by default,
            # since a general A diag(w) A^T need not be an M-matrix).
            self.kind = "sddm"
            self.assembler = CSCAssembler(polytope.A)
            self._options = self._resolve_options(opts)
            self.method = self._options.get("method")
            self.phase = None

    @staticmethod
    def _resolve_options(options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in a safe default ``method`` for general SPD systems.

        Args:
            options: User-supplied LaplacianSolve options.

        Returns:
            The options with ``method`` defaulted to ``"cholmod"`` when the
            CHOLMOD backend is available and the user did not specify one.

        """
        opts = dict(options)
        if "method" not in opts:
            try:
                from purc.laplaciansolve._loader import has_cholmod

                opts["method"] = "cholmod" if has_cholmod() else "auto"
            except Exception:  # pragma: no cover - defensive
                opts["method"] = "auto"
        return opts

    def solve(self, w: torch.Tensor, eps: float, rhs: torch.Tensor) -> torch.Tensor:
        """
        Solve ``(A diag(w) A^T + eps I) d = rhs`` for one right-hand side.

        Weights and RHS are torch tensors; they bridge to NumPy zero-copy on CPU
        for the SciPy assembly and the LaplacianSolve call, and the solution is
        returned as a torch tensor matching the solver's dtype.

        Args:
            w: Per-coordinate weights ``D * active_mask``, shape ``(N,)``.
            eps: Diagonal regularizer (``> 0`` so the system is SPD).
            rhs: Right-hand side, shape ``(k,)``.

        Returns:
            The Newton step ``d`` as a torch tensor, shape ``(k,)``.

        """
        if self.kind == "purc":
            # One GIL-released native call: assemble C diag(w) C^T + eps I and
            # solve.  Inactive edges carry weight 0 (folded by the SSN solver).
            res = self._purc.solve_step(to_numpy(w), eps=float(eps), rhs=to_numpy(rhs))
            return as_tensor(res["solution"])

        from purc.laplaciansolve import SDDMSolver, SolverConfig

        M = self.assembler.assemble(to_numpy(w), float(eps))
        if self._solver is None:
            self._solver = SDDMSolver(M, config=SolverConfig(**self._options))
        else:
            # Same pattern across iterations -> reuse the symbolic factorization.
            self._solver.update(M)
        # LaplacianSolve accepts a CPU torch tensor (zero-copy through its numpy
        # buffer) and restores the solution to the same type/dtype/device.
        sol = self._solver.solve(as_tensor(rhs))
        return as_tensor(sol)

    def prepare(self, w: torch.Tensor, eps: float) -> None:
        """
        Factorize ``A diag(w) A^T + eps I`` once for reuse across right-hand sides.

        The Mehrotra predictor-corrector IPM solves two systems with the *same*
        matrix (predictor and corrector) but different RHS per iteration; calling
        :meth:`prepare` once then :meth:`solve_prepared` twice does a single
        numeric factorization instead of two.

        Args:
            w: Per-coordinate weights, shape ``(N,)``.
            eps: Diagonal regularizer (``> 0``).

        """
        if self.kind == "purc":
            # PURCLaplacianSolver fuses assembly+factor+solve in one native call and
            # exposes no separate factor phase; cache (w, eps) and let
            # solve_prepared re-run the fused call.  The network fast path (forest /
            # already-cheap CHOLMOD) is the path where this matters least.
            self._prep = (to_numpy(w), float(eps))
            return
        from purc.laplaciansolve import SDDMSolver, SolverConfig

        M = self.assembler.assemble(to_numpy(w), float(eps))
        if self._solver is None:
            self._solver = SDDMSolver(M, config=SolverConfig(**self._options))
        else:
            # Same sparsity pattern across iterations -> reuse the symbolic phase,
            # redo only the numeric factorization.
            self._solver.update(M)

    def solve_prepared(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        Solve against the factorization built by the last :meth:`prepare`.

        Args:
            rhs: Right-hand side, shape ``(k,)``.

        Returns:
            The Newton step ``d`` as a torch tensor, shape ``(k,)``.

        Raises:
            RuntimeError: If called before :meth:`prepare`.

        """
        if self.kind == "purc":
            if self._prep is None:
                raise RuntimeError("call prepare(w, eps) before solve_prepared(rhs)")
            w, eps = self._prep
            res = self._purc.solve_step(w, eps=eps, rhs=to_numpy(rhs))
            return as_tensor(res["solution"])
        if self._solver is None:
            raise RuntimeError("call prepare(w, eps) before solve_prepared(rhs)")
        return as_tensor(self._solver.solve(as_tensor(rhs)))

    def solve_batch(
        self, weights: torch.Tensor, eps: torch.Tensor, rhs: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve a batch of Newton systems (one per OD-pair) in one shot.

        For the incidence path this is a single GIL-released
        ``PURCLaplacianSolver.solve_batch`` call.  For the general path the
        per-system CSC values (all sharing the cached pattern) are stacked and
        handed to a single GIL-released ``BatchedSDDMSolver.solve_batch`` -- one
        batched factorization+solve over the whole OD batch, not a Python loop of
        per-system factorizations.

        Args:
            weights: Per-system per-coordinate weights ``[B, N]``.
            eps: Per-system regularizers ``[B]``.
            rhs: Per-system right-hand sides ``[B, k]``.

        Returns:
            The Newton steps ``[B, k]`` as a torch tensor.

        """
        if self.kind == "purc":
            res = self._purc.solve_batch(to_numpy(weights), eps=to_numpy(eps), rhs=to_numpy(rhs))
            return as_tensor(res["solution"])
        # General path: assemble per-system values at the fixed pattern, then one
        # native batched solve.  M_b = A diag(w_b) A^T + eps_b I is SPD (eps_b > 0),
        # so no grounding is needed and every system shares the assembler's pattern.
        w_np = to_numpy(weights)  # [B, N]
        eps_np = to_numpy(eps)  # [B]
        rhs_np = to_numpy(rhs)  # [B, k]
        values = self.assembler.assemble_values_batch(w_np, eps_np)  # [B, nnz], shared pattern
        if self._batched is None:
            from purc.laplaciansolve import BatchedSDDMSolver, SolverConfig

            m0 = self.assembler.assemble(w_np[0], float(eps_np[0]))
            self._batched = BatchedSDDMSolver(m0, config=SolverConfig(**self._options))
        sol = self._batched.solve_batch(values, rhs_np)  # [B, k]
        return as_tensor(sol)
