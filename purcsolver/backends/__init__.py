"""
Backend routing for the per-Newton-iteration linear solve.

The Newton system ``(A_S diag(D) A_S^T + eps I) d = -r`` has a fixed sparsity
pattern across iterations, so the LaplacianSolve handle is built once and reused.
``routing.py`` (M1/M4) dispatches incidence constraints to ``PURCLaplacianSolver``
and general constraints to ``BatchedSDDMSolver`` / ``SDDMSolver``; ``assembly.py``
(M1) maps the per-coordinate weights ``D`` to CSC values for the general path.
M0 ships this package as a placeholder.
"""

from __future__ import annotations

__all__: list[str] = []
