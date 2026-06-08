"""
Exact zero-fill forest-LDL solver (the acyclic fast path for SDDM systems).

When the active subgraph of the PURC matrix ``M = H + eps*I`` is a forest, ``M``
factorizes with no fill-in and solves in O(n) by leaf elimination - faster than a
general sparse Cholesky, with no ordering search and no library overhead.  The
native :class:`ForestSolver` *detects* the forest in the same leaf-pruning pass
that builds the factorization, so :attr:`is_forest` is available right after
construction for the auto-select router to act on.
"""

from __future__ import annotations

import numpy as np

from ._direct_base import _DirectSDDMBase
from ._loader import native_core


class ForestLDLSDDM(_DirectSDDMBase):
    """Reusable exact solver for a fixed-pattern SDDM matrix with a forest graph."""

    def __init__(self, m, *, config=None) -> None:
        """
        Detect the forest and, if acyclic, build the exact zero-fill LDL of *m*.

        Args:
            m: A symmetric SDDM matrix whose off-diagonal graph may be a forest
                (scipy / numpy / torch; CPU).
            config: Accepted for interface symmetry with the other handles.

        """
        self.config = config
        csc = self._setup_pattern(m)
        self._solver = native_core().ForestSolver(
            self._indptr, self._indices, csc.data.astype(np.float64), self._n
        )
        self.is_forest = bool(self._solver.is_forest())
