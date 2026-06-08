r"""
The unified PURC forward solver: one entry point, a provably-ruled regime.

There is no single *uniform* iteration that is simultaneously fast and robust on
every kernel -- and the reason is structural, not incidental:

* For a kernel that is **smooth on the closed box** (quadratic, polynomial
  sieve, modified entropy), the solution **saturates** the box, so the dual
  semismooth Newton method must identify a combinatorial active set and
  *thrashes* on large cold starts (on Chicago Regional the cold SSN fails
  outright).  The **primal-dual interior-point method** smooths the box with a
  barrier and converges in ~15 Laplacian solves with no active-set combinatorics.

* For a **Legendre-type** kernel (Shannon / logit entropy), ``h'`` diverges at a
  box face, so a primal box barrier is ill posed
  (:meth:`SeparablePerturbation.admits_primal_interior` is ``False``) -- but the
  same divergence makes that face *provably never active*, so the solution is
  essentially interior and the **dual semismooth Newton method** converges in
  ~20 iterations with no thrashing (and an order of magnitude faster than CVXPY).

So the regime is selected by a **provable property of the (kernel, box)** -- the
finiteness of ``h'`` on the closed box -- evaluated once at
:meth:`preprocess`.  This is a regime change governed by a theorem, not a runtime
failure caught mid-iteration.  As a universal safety net, the barrier-smoothed
dual continuation (:class:`BarrierContinuationSolver`), which is provably
convergent for *every* strictly-convex kernel, is used as a fallback if the
fast-path engine ever reports non-convergence.

From the caller's perspective this is a single algorithm: ``preprocess`` then
``solve``, no method to choose, no parameter to tune.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..config import ForwardSolverConfig
from ..problem import PUMProblem
from ..result import PURCResult
from ..utils.logging import get_logger
from ..utils.typing import ArrayLike
from . import SOLVERS
from .barrier_ipm import BarrierContinuationSolver
from .base import ForwardSolver
from .ipm import IPMSolver
from .ssn import RegularizedSSNSolver

_logger = get_logger(__name__)


class AutoSolver(ForwardSolver):
    """
    Unified forward solver with a provably-ruled fast-path regime + robust backstop.

    The regime is chosen at :meth:`preprocess` from
    :meth:`SeparablePerturbation.admits_primal_interior`:

    * ``True`` (smooth on the closed box) -> :class:`IPMSolver` (primal-dual IPM
      + semismooth-Newton crossover);
    * ``False`` (Legendre type at a box face) -> :class:`RegularizedSSNSolver`.

    If the fast-path engine returns ``success=False`` on a given right-hand side,
    the solve is retried with the provably-convergent
    :class:`BarrierContinuationSolver` and that result is returned instead.

    Args:
        config: Solver configuration shared by the underlying engines.

    """

    def __init__(self, config: Optional[ForwardSolverConfig] = None) -> None:
        super().__init__(config)
        self._engine: Optional[ForwardSolver] = None
        self._fallback: Optional[BarrierContinuationSolver] = None
        self.regime: Optional[str] = None

    def preprocess(self, problem: PUMProblem) -> None:
        """
        Select the regime by the provable rule and build the fast-path engine.

        Args:
            problem: The forward problem whose (kernel, box) fixes the regime.

        """
        self._problem = problem
        c = problem.constraint
        if problem.perturbation.admits_primal_interior(c.lo, c.hi):
            self._engine = IPMSolver(self.config)
            self.regime = "ipm"
        else:
            self._engine = RegularizedSSNSolver(self.config)
            self.regime = "ssn"
        self._engine.preprocess(problem)

    def solve(
        self,
        theta: Tuple[ArrayLike, ArrayLike],
        *,
        b: Optional[ArrayLike] = None,
        lam0: Optional[ArrayLike] = None,
    ) -> PURCResult:
        """
        Solve at ``theta`` with the regime engine, falling back if it does not converge.

        Args:
            theta: ``(beta, gamma)``.
            b: Optional equality right-hand side override.
            lam0: Optional warm-start multipliers.

        Returns:
            The forward-solve result, tagged with ``extras["regime"]``.

        Raises:
            RuntimeError: If :meth:`preprocess` has not been called.

        """
        if self._engine is None or self._problem is None:
            raise RuntimeError("call preprocess(problem) before solve()")
        res = self._engine.solve(theta, b=b, lam0=lam0)
        if not res.success:
            res = self._run_fallback(theta, b=b, lam0=lam0)
        res.extras.setdefault("regime", self.regime)
        return res

    def solve_batch(
        self,
        theta: Tuple[ArrayLike, ArrayLike],
        b_batch: ArrayLike,
        *,
        lam0: Optional[ArrayLike] = None,
    ) -> PURCResult:
        """
        Solve a batch of OD-pairs with the regime engine.

        Routes to the same engine chosen at :meth:`preprocess` -- the batched IPM
        for box-saturating kernels, the batched dual SSN for Legendre kernels --
        each of which vectorizes the solve over the ``B`` systems natively.

        Args:
            theta: ``(beta, gamma)`` shared by all systems.
            b_batch: Per-system demands, shape ``[B, k]``.
            lam0: Optional warm-start multipliers ``[B, k]``.

        Returns:
            The batched forward-solve result, tagged with ``extras["regime"]``.

        Raises:
            RuntimeError: If :meth:`preprocess` has not been called.

        """
        if self._engine is None or self._problem is None:
            raise RuntimeError("call preprocess(problem) before solve_batch()")
        res = self._engine.solve_batch(theta, b_batch, lam0=lam0)
        res.extras.setdefault("regime", self.regime)
        return res

    def _run_fallback(
        self,
        theta: Tuple[ArrayLike, ArrayLike],
        *,
        b: Optional[ArrayLike],
        lam0: Optional[ArrayLike],
    ) -> PURCResult:
        """
        Solve with the provably-convergent barrier continuation (lazy build).

        Args:
            theta: ``(beta, gamma)``.
            b: Optional equality right-hand side override.
            lam0: Optional warm-start multipliers.

        Returns:
            The barrier-continuation result, tagged ``extras["regime"] = "barrier(fallback)"``.

        """
        _logger.warning(
            "%s fast path did not converge; falling back to barrier continuation",
            self.regime,
        )
        if self._fallback is None:
            self._fallback = BarrierContinuationSolver(self.config)
            self._fallback.preprocess(self._problem)
        res = self._fallback.solve(theta, b=b, lam0=lam0)
        res.extras["regime"] = "barrier(fallback)"
        return res


SOLVERS["auto"] = AutoSolver
