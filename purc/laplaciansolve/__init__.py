"""
LaplacianSolve: fast Laplacian / SDDM linear solvers (approxChol port).

Stage 0 exposes the pure-Python reference oracle under
:mod:`purc.laplaciansolve.reference`.  Later stages add the C++/CUDA backends and the
PURC-specialized batched API behind the same public names.
"""

from __future__ import annotations

from .logging import configure_logging, get_logger, logger

__version__ = "0.0.1"


def __getattr__(name: str):
    """
    Lazily expose the native-backed solver handles (avoids import cost / hard native dep).

    Args:
        name: Attribute name being accessed.

    Returns:
        The requested solver class.

    Raises:
        AttributeError: If *name* is not a public attribute.

    """
    if name in ("LaplacianSolver", "SDDMSolver", "SolverConfig"):
        from .solver import LaplacianSolver, SDDMSolver, SolverConfig

        return {
            "LaplacianSolver": LaplacianSolver,
            "SDDMSolver": SDDMSolver,
            "SolverConfig": SolverConfig,
        }[name]
    if name == "BatchedSDDMSolver":
        from .batched import BatchedSDDMSolver

        return BatchedSDDMSolver
    if name == "PURCLaplacianSolver":
        from .purc import PURCLaplacianSolver

        return PURCLaplacianSolver
    if name == "solve":
        from .dispatch import solve

        return solve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "logger",
    "get_logger",
    "configure_logging",
    "LaplacianSolver",
    "SDDMSolver",
    "BatchedSDDMSolver",
    "PURCLaplacianSolver",
    "SolverConfig",
    "solve",
]
