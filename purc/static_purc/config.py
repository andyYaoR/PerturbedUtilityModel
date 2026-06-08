"""
Solver configuration for the regularized semismooth Newton (SSN) method.

The dual SSN loop (see the package README) repeatedly:
  1. recovers the primal ``x_hat`` from the current multipliers ``lambda``,
  2. forms the residual ``r = b - A x_hat`` (the dual gradient),
  3. solves a regularized Newton system ``(H + eps_k I) d = -r`` with
     ``eps_k = clip(min(eps0, ||r||), eps_floor, eps0)``, and
  4. takes an Armijo backtracking step on the convex dual objective.

:class:`SSNConfig` collects the knobs for that loop plus the inner-linear-solve
config that is forwarded verbatim to LaplacianSolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SSNConfig:
    """
    Configuration for :class:`purcsolver.solvers.ssn.RegularizedSSNSolver`.

    Attributes:
        tol: Convergence tolerance on the dual-gradient residual ``||r||_inf``
            (the KKT primal-feasibility residual ``||A x_hat - b||_inf``).
        max_iter: Maximum number of semismooth Newton iterations.
        eps0: Initial / maximum Newton regularization ``eps_0``.  The per-step
            regularizer is ``eps_k = min(eps0, ||r||)`` (2-norm), floored below.
            Keep it small: it only needs to make ``H + eps_k I`` SPD (the active
            Hessian is low-rank for sparse single-OD problems).  A large ``eps0``
            over-damps the Newton step and destroys the quadratic convergence
            (e.g. on hard instances ``1e-2`` stalls where ``1e-7`` converges in
            ~15 iterations).
        eps_floor: Hard lower bound on ``eps_k``.  Keeps the Newton system
            solvable when the active set is empty/degenerate (``H = 0``), where
            ``eps_k I`` is doing double duty as regularizer and nullspace cover.
        armijo_c1: Armijo sufficient-decrease parameter (``0 < c1 < 1/2``).
        armijo_beta: Armijo backtracking shrink factor in ``(0, 1)``.
        max_linesearch: Maximum backtracking steps per Newton iteration (legacy
            Armijo path).
        lm_eps0: Cold-start Levenberg-Marquardt damping (warm-started thereafter).
            Optimistic and small: well-conditioned solves stay near-Newton, while
            the empty-active-set start needs only a few damping increases.
        lm_max_tries: Maximum damping adjustments within one Newton iteration.
        stall_patience: Stop early if the residual fails to improve by more than
            a tiny relative amount for this many consecutive iterations *and* it
            is already below :attr:`stall_floor` (i.e. genuinely at the numerical
            floor); avoids spinning to ``max_iter`` without prematurely stopping a
            slowly-but-truly-converging solve.
        stall_floor: Residual level below which a plateau is treated as the
            numerical floor.  The dual gradient ``r = A x_hat - b`` is non-monotone
            (the line search decreases the dual objective, not ``r``), so a plateau
            at a *large* ``r`` must not trigger an early stop.
        warm_start: If ``True``, persist the solution multipliers between
            successive ``solve`` calls so a sweep over slightly varying ``theta``
            (the estimation outer loop) converges in few iterations.
        laplacian: Options forwarded to the LaplacianSolve ``SolverConfig`` (e.g.
            ``{"tol": 1e-10, "method": "auto"}``).  Kept as a plain dict so this
            package does not import LaplacianSolve at config-construction time.
        profile: Emit per-iteration timing/diagnostic logs when ``True``.

    """

    tol: float = 1e-9
    max_iter: int = 100
    eps0: float = 1e-7
    eps_floor: float = 1e-12
    armijo_c1: float = 1e-4
    armijo_beta: float = 0.5
    max_linesearch: int = 30
    stall_patience: int = 8
    stall_floor: float = 1e-6
    lm_eps0: float = 1e-2
    lm_max_tries: int = 40
    warm_start: bool = True
    laplacian: Dict[str, Any] = field(default_factory=dict)
    profile: bool = False

    def __post_init__(self) -> None:
        """
        Validate the configuration on construction.

        Raises:
            ValueError: If any parameter is outside its admissible range.

        """
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.eps0 <= 0:
            raise ValueError(f"eps0 must be > 0, got {self.eps0}")
        if not (0 < self.eps_floor <= self.eps0):
            raise ValueError(
                f"eps_floor must satisfy 0 < eps_floor <= eps0, got "
                f"eps_floor={self.eps_floor}, eps0={self.eps0}"
            )
        if not (0 < self.armijo_c1 < 0.5):
            raise ValueError(f"armijo_c1 must be in (0, 1/2), got {self.armijo_c1}")
        if not (0 < self.armijo_beta < 1):
            raise ValueError(f"armijo_beta must be in (0, 1), got {self.armijo_beta}")
        if self.max_linesearch < 1:
            raise ValueError(f"max_linesearch must be >= 1, got {self.max_linesearch}")
        if self.stall_patience < 1:
            raise ValueError(f"stall_patience must be >= 1, got {self.stall_patience}")
        if self.lm_eps0 <= 0:
            raise ValueError(f"lm_eps0 must be > 0, got {self.lm_eps0}")
        if self.lm_max_tries < 1:
            raise ValueError(f"lm_max_tries must be >= 1, got {self.lm_max_tries}")

    def laplacian_config(self) -> Optional[Any]:
        """
        Build a LaplacianSolve ``SolverConfig`` from :attr:`laplacian`.

        Imported lazily so the native LaplacianSolve dependency is only required
        when a solve actually runs, not when an :class:`SSNConfig` is created.

        Returns:
            A ``purc.laplaciansolve.SolverConfig`` instance, or ``None`` if
            LaplacianSolve is unavailable (callers may then use a fallback).

        """
        try:
            from purc.laplaciansolve import SolverConfig
        except ImportError:  # pragma: no cover - exercised only without the dep
            return None
        return SolverConfig(**self.laplacian)
