r"""
Abstract base class for forward solvers.

A forward solver minimizes ``F(x;gamma) - v(beta)^T x`` over the constraint
polytope.  The interface is deliberately split into a build-once
:meth:`preprocess` and a cheap, repeatable :meth:`solve` so the estimation outer
loop -- which sweeps slightly varying ``theta = (beta, gamma)`` across many
right-hand sides -- reuses the persistent factorization/symbolic pattern and the
warm-started multipliers.

The solver returns a :class:`~purc.static_purc.result.PURCResult` carrying the primal
``x*`` and the conjugate value ``F*``; by the Fenchel-Young envelope property
those are everything the estimation gradient needs, so no ``dx*/dtheta``
sensitivity machinery lives in the solver core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ..config import ForwardSolverConfig
from ..problem import PUMProblem
from ..result import PURCResult
from ..utils.typing import ArrayLike


class ForwardSolver(ABC):
    """
    Interface for a perturbed-utility forward solver.

    Args:
        config: Solver configuration (tolerances, regularization, line search).

    """

    def __init__(self, config: Optional[ForwardSolverConfig] = None) -> None:
        self.config = config or ForwardSolverConfig()
        self._problem: Optional[PUMProblem] = None

    @abstractmethod
    def preprocess(self, problem: PUMProblem) -> None:
        """
        Build the persistent solver state for ``problem`` (called once).

        Implementations construct the LaplacianSolve backend handle from the
        fixed sparsity pattern, the perturbation kernel descriptor, and the
        multiplier-normalization map here, so subsequent :meth:`solve` calls only
        ship the changing weights / right-hand side.

        Args:
            problem: The forward problem whose structure is fixed across solves.

        """

    @abstractmethod
    def solve(
        self,
        theta: Tuple[ArrayLike, ArrayLike],
        *,
        b: Optional[ArrayLike] = None,
        lam0: Optional[ArrayLike] = None,
    ) -> PURCResult:
        """
        Solve the forward problem at parameters ``theta``.

        Args:
            theta: Parameters ``(beta, gamma)``.  ``beta`` sets the linear utility
                ``v = Z @ beta``; ``gamma`` are the perturbation shape parameters.
            b: Optional override of the equality right-hand side (or a batch of
                them, shape ``(B, k)``) for multi-OD solves.  Defaults to the
                constraint's own ``b``.
            lam0: Optional warm-start multipliers.  Defaults to the persisted
                solution from the previous call when ``config.warm_start``.

        Returns:
            A :class:`PURCResult` with the primal ``x*``, multipliers, and the
            conjugate value ``F*``.

        """
