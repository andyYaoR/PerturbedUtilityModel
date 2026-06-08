r"""
CVXPY reference oracle.

Builds the equivalent disciplined-convex program for a perturbation that exposes
a :meth:`~purc.static_purc.perturbations.base.SeparablePerturbation.cvxpy_h` expression
(quadratic, entropy, modified entropy, ...) and solves it with a high-accuracy
conic solver.  Its solution is an independent ground truth for the SSN solver --
independent because it shares no code path with the semismooth Newton iteration.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..problem import PUMProblem
from ..utils.torch_compat import to_numpy
from ..utils.typing import ArrayLike


def solve_cvxpy(
    problem: PUMProblem,
    theta: Tuple[ArrayLike, ArrayLike],
    *,
    solver: Optional[str] = None,
    **solve_kwargs: Any,
) -> np.ndarray:
    """
    Solve ``min_x sum_i ell_i h(x_i) - v^T x`` over the polytope with CVXPY.

    Args:
        problem: The forward problem (perturbation + constraint + utility map).
        theta: Parameters ``(beta, gamma)``.
        solver: Optional CVXPY solver name (default Clarabel for accuracy).
        **solve_kwargs: Extra keyword arguments forwarded to ``Problem.solve``.

    Returns:
        The optimal primal ``x*``, shape ``(N,)``.

    Raises:
        RuntimeError: If CVXPY does not reach an optimal status.

    """
    import cvxpy as cp

    c = problem.constraint
    pert = problem.perturbation
    beta, gamma = theta
    # Bridge all torch data to numpy for the (numpy-based) cvxpy model.
    v = to_numpy(problem.utility(beta))
    ell = to_numpy(c.ell)
    b_np = to_numpy(c.b)
    lo_np = to_numpy(c.lo)
    hi_np = to_numpy(c.hi)

    x = cp.Variable(c.num_coords)
    h_expr = pert.cvxpy_h(x, gamma)
    objective = cp.Minimize(cp.sum(cp.multiply(ell, h_expr)) - v @ x)
    constraints = [c.A @ x == b_np, x >= lo_np, x <= hi_np]
    prob = cp.Problem(objective, constraints)

    if solver is None:
        # Clarabel's defaults give only ~1e-6 primal accuracy; tighten so the
        # oracle is a sharp ground truth for the SSN solver (which hits ~1e-11).
        solver = cp.CLARABEL
        solve_kwargs.setdefault("tol_gap_abs", 1e-11)
        solve_kwargs.setdefault("tol_gap_rel", 1e-11)
        solve_kwargs.setdefault("tol_feas", 1e-11)
    prob.solve(solver=solver, **solve_kwargs)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CVXPY did not converge: status={prob.status}")
    return np.asarray(x.value, dtype=float)
