"""
Reference oracles for correctness and performance cross-checks.

``cvxpy_oracle.py`` (M1) builds the equivalent DCP convex program for
perturbations that CVXPY can express (quadratic, entropy, Tsallis).
``scipy_oracle.py`` (M1) is an independent separable solver (trust-constr plus a
dense dual active-set Newton) used as ground truth for the polynomial sieve and
as a cross-check elsewhere; it deliberately does *not* call LaplacianSolve so it
cannot co-validate solver bugs.  M0 ships this package as a placeholder.
"""

from __future__ import annotations

from typing import Tuple

from ..problem import PUMProblem
from ..utils.typing import ArrayLike
from .cvxpy_oracle import solve_cvxpy
from .scipy_oracle import solve_scipy


def solve_oracle(problem: PUMProblem, theta: Tuple[ArrayLike, ArrayLike], **kwargs):
    """
    Solve with the best available reference: CVXPY when the perturbation is
    DCP-expressible, else the independent dense scipy dual-Newton oracle.

    Args:
        problem: The forward problem.
        theta: Parameters ``(beta, gamma)``.
        **kwargs: Forwarded to the chosen oracle.

    Returns:
        The optimal primal ``x*``.

    """
    if problem.perturbation.cvxpy_expressible:
        return solve_cvxpy(problem, theta, **kwargs)
    return solve_scipy(problem, theta, **kwargs)


__all__ = ["solve_cvxpy", "solve_scipy", "solve_oracle"]
