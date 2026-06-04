"""
Registry of forward solvers.

M0 ships the :class:`ForwardSolver` ABC only.  The regularized semismooth Newton
solver (:class:`RegularizedSSNSolver`) lands in M1.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import ForwardSolver

# name -> ForwardSolver subclass
SOLVERS: Dict[str, Type[ForwardSolver]] = {}


def get_solver(name: str, **kwargs: Any) -> ForwardSolver:
    """
    Instantiate a registered forward solver by name.

    Args:
        name: Registry key (e.g. ``"ssn"``).
        **kwargs: Forwarded to the solver constructor.

    Returns:
        A new solver instance.

    Raises:
        KeyError: If ``name`` is not registered.

    """
    if name not in SOLVERS:
        raise KeyError(f"unknown solver {name!r}; registered: {sorted(SOLVERS)}")
    return SOLVERS[name](**kwargs)


__all__ = [
    "ForwardSolver",
    "SOLVERS",
    "get_solver",
    "RegularizedSSNSolver",
    "IPMSolver",
    "BarrierContinuationSolver",
    "AutoSolver",
]

# Import concrete solvers for their registration side effects (kept at the
# bottom so SOLVERS / ForwardSolver are defined before the solvers import them).
from .ssn import RegularizedSSNSolver  # noqa: E402
from .ipm import IPMSolver  # noqa: E402
from .barrier_ipm import BarrierContinuationSolver  # noqa: E402
from .auto import AutoSolver  # noqa: E402
