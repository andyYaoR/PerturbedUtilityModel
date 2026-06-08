"""
PURCSolver: perturbed utility models with general polytope constraints.

Solves ``min_x F(x;gamma) - v(beta)^T x`` over a polytope ``{Ax=b, l<=x<=u}`` with
a strictly-convex separable perturbation ``F``, via a regularized semismooth
Newton method whose per-iteration linear solve is delegated to the optimized,
GIL-released LaplacianSolve backend.

The public surface is exposed lazily (so importing the top-level package does not
force the native core or the optional cvxpy oracle to load).  M0 ships the core
dataclasses, the registries, and the abstract interfaces; concrete perturbations,
constraints, and the SSN solver are added in M1+.
"""

from __future__ import annotations

from .config import ForwardSolverConfig
from .problem import PUMProblem
from .result import PURCResult
from .utils.logging import configure_logging, get_logger, logger
from .utils.native import native_available

__version__ = "0.0.0"


def __getattr__(name: str):
    """
    Lazily expose registries and base classes to avoid import-time cost.

    Args:
        name: Attribute name being accessed.

    Returns:
        The requested public object.

    Raises:
        AttributeError: If *name* is not a public attribute.

    """
    if name in ("SeparablePerturbation", "PERTURBATIONS", "get_perturbation"):
        from . import perturbations

        return getattr(perturbations, name)
    if name == "Polytope":
        from .constraints import Polytope

        return Polytope
    if name in ("ForwardSolver", "SOLVERS", "get_solver"):
        from . import solvers

        return getattr(solvers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "ForwardSolverConfig",
    "PUMProblem",
    "PURCResult",
    "SeparablePerturbation",
    "Polytope",
    "ForwardSolver",
    "PERTURBATIONS",
    "SOLVERS",
    "get_perturbation",
    "get_solver",
    "native_available",
    "logger",
    "get_logger",
    "configure_logging",
]
