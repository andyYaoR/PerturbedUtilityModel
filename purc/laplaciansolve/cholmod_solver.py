"""
CHOLMOD-backed direct SPD solver (a fast path for the PURC matrix H + eps*I).

Wraps the native ``CholmodSolver`` with the shared array-backend I/O in
:class:`~purc.laplaciansolve._direct_base._DirectSDDMBase` (numpy / scipy / torch).
``M = H + eps*I`` is SPD, so it is factorized directly - no grounding, no PCG -
matching (and, batched, beating) Julia's ``cholesky``.  The symbolic
factorization is reused across weight/eps updates (numeric-only refactorization)
and across many right-hand sides.
"""

from __future__ import annotations

import numpy as np

from ._direct_base import _DirectSDDMBase
from ._loader import cholmod_core


class CholmodSDDM(_DirectSDDMBase):
    """Reusable direct solver for a fixed-pattern SPD matrix via CHOLMOD."""

    def __init__(self, m, *, config=None) -> None:
        """
        Analyze + factorize the SPD matrix *m*.

        Args:
            m: A symmetric positive-definite matrix (scipy / numpy / torch; CPU).
            config: Accepted for interface symmetry with the other handles.

        """
        self.config = config
        csc = self._setup_pattern(m)
        self._solver = cholmod_core().CholmodSolver(
            self._indptr, self._indices, csc.data.astype(np.float64), self._n
        )
