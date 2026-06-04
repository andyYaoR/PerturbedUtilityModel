r"""
Backend dispatcher for the per-Newton-iteration linear solve.

The Newton system ``(A diag(w) A^T + eps I) d = rhs`` has a fixed sparsity
pattern, so the LaplacianSolve handle is built once and reused across iterations
-- only ``w``, ``eps`` and ``rhs`` change.  :class:`LaplacianBackend` builds the
right LaplacianSolve solver for the constraint geometry:

* **general A** (this milestone): assemble ``M`` via :class:`CSCAssembler` and
  drive ``SDDMSolver``.  Because a general ``A diag(w) A^T`` is SPD but *not*
  necessarily an M-matrix (off-diagonals may be positive), we force
  ``method="cholmod"`` (a true general SPD Cholesky) when CHOLMOD is available;
  the forest / approxChol routes assume Laplacian structure and would misroute.
* **incidence A** (v0.4.0): route to ``PURCLaplacianSolver`` for the network fast
  path.

The assembler and the fixed pattern are the parts ported to the native core in
v0.3.0; this Python path is the reference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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
        laplacian_options: Options forwarded to ``laplaciansolve.SolverConfig``
            (e.g. ``{"method": "cholmod", "tol": 1e-10}``).  ``method`` defaults
            to ``"cholmod"`` when available, else ``"auto"``.

    """

    def __init__(
        self,
        polytope: Polytope,
        laplacian_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if polytope.is_incidence:
            # The incidence/network fast path (PURCLaplacianSolver) lands in
            # v0.4.0; for now incidence matrices go through the general path too.
            _logger.debug("incidence fast path not yet enabled; using general SDDM path")
        self.assembler = CSCAssembler(polytope.A)
        self.k = polytope.num_constraints
        self._options = self._resolve_options(laplacian_options or {})
        self._solver = None  # built lazily on first solve (needs a sample M)

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
                from laplaciansolve._loader import has_cholmod

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
        from laplaciansolve import SDDMSolver, SolverConfig

        M = self.assembler.assemble(to_numpy(w), float(eps))
        if self._solver is None:
            self._solver = SDDMSolver(M, config=SolverConfig(**self._options))
        else:
            # Same pattern across iterations -> reuse the symbolic factorization.
            self._solver.update(M)
        sol = self._solver.solve(to_numpy(rhs))
        return as_tensor(sol)
