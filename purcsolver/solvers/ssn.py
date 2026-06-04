r"""
Regularized semismooth Newton (SSN) solver for the perturbed-utility forward
problem on its convex dual.

For ``min_x F(x;gamma) - v^T x`` over ``X = {Ax=b, lo<=x<=hi}`` with separable
``F(x) = sum_i ell_i h(x_i)``, the dual (one multiplier ``lambda`` per equality
row) is the smooth convex program

    min_lambda  phi(lambda) = -b^T lambda + sum_i ell_i h*(eta_i),
    eta_i = (v_i + (A^T lambda)_i) / ell_i,

with closed-form primal recovery ``x_hat_i = xi*(eta_i)`` (clipped to the box),
gradient ``grad phi = A x_hat - b =: r``, and generalized Hessian
``H = A diag(D) A^T`` where ``D_i = 1/(ell_i h''(x_hat_i))`` on the active set
``{lo < x_hat_i < hi}`` and ``0`` at saturated coordinates.

Each iteration solves the regularized Newton system ``(H + eps_k I) d = -r`` (the
linear solve is delegated to LaplacianSolve via :class:`LaplacianBackend`) with
``eps_k = clip(min(eps0, ||r||), eps_floor, eps0)`` -- which both makes the
system SPD and bounds the step along ``H``'s nullspace, so no separate multiplier
gauge-fixing is needed -- followed by an Armijo line search on ``phi``.  Strong
semismoothness gives local Q-quadratic convergence; the line search globalizes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..backends.routing import LaplacianBackend
from ..config import SSNConfig
from ..problem import PUMProblem
from ..result import (
    STATUS_CONVERGED,
    STATUS_LINESEARCH_FAILED,
    STATUS_MAX_ITER,
    PURCResult,
)
from ..utils.logging import get_logger
from ..utils.typing import ArrayLike
from . import SOLVERS
from ._linesearch import armijo_backtracking
from .base import ForwardSolver

_logger = get_logger(__name__)


class RegularizedSSNSolver(ForwardSolver):
    """
    Regularized semismooth Newton solver (the package's primary forward solver).

    Build the persistent backend with :meth:`preprocess`, then call :meth:`solve`
    repeatedly with varying ``theta`` / ``b`` (warm-started by default).
    """

    def __init__(self, config: Optional[SSNConfig] = None) -> None:
        super().__init__(config)
        self._backend: Optional[LaplacianBackend] = None
        self._lam: Optional[np.ndarray] = None

    def preprocess(self, problem: PUMProblem) -> None:
        """
        Build the persistent LaplacianSolve backend from the fixed pattern.

        Args:
            problem: The forward problem whose constraint structure is fixed.

        """
        self._problem = problem
        self._backend = LaplacianBackend(problem.constraint, self.config.laplacian)
        self._lam = np.zeros(problem.constraint.num_constraints, dtype=float)

    # -- internals ----------------------------------------------------------

    def _recover(
        self, v: ArrayLike, lam: ArrayLike, gamma: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map multipliers to ``(eta, x_hat, interior_mask)``.

        Args:
            v: Link utilities ``v(beta)``.
            lam: Current multipliers.
            gamma: Perturbation parameters.

        Returns:
            ``(eta, x_hat, interior_mask)``.

        """
        c = self._problem.constraint
        pert = self._problem.perturbation
        eta = (v + c.rmatvec(lam)) / c.ell
        x_hat, interior = pert.primal_recovery(eta, c.lo, c.hi, gamma)
        return eta, x_hat, interior

    def _phi(self, v: ArrayLike, lam: ArrayLike, b: ArrayLike, gamma: ArrayLike) -> float:
        """
        Evaluate the dual objective ``phi(lambda)``.

        Args:
            v: Link utilities.
            lam: Multipliers.
            b: Equality right-hand side.
            gamma: Perturbation parameters.

        Returns:
            The scalar ``phi(lambda)``.

        """
        c = self._problem.constraint
        pert = self._problem.perturbation
        eta = (v + c.rmatvec(lam)) / c.ell
        conj = pert.conj_box(eta, c.lo, c.hi, gamma)
        return float(-b @ lam + c.ell @ conj)

    def solve(
        self,
        theta: Tuple[ArrayLike, ArrayLike],
        *,
        b: Optional[ArrayLike] = None,
        lam0: Optional[ArrayLike] = None,
    ) -> PURCResult:
        """
        Solve the forward problem at parameters ``theta = (beta, gamma)``.

        Args:
            theta: ``(beta, gamma)``; ``beta`` sets ``v = Z beta``, ``gamma`` the
                perturbation shape.
            b: Optional equality right-hand side override, shape ``(k,)``.
            lam0: Optional warm-start multipliers; defaults to the persisted
                solution (or zeros) when ``config.warm_start``.

        Returns:
            The forward-solve result carrying the primal ``x*``, the multipliers,
            and the Fenchel conjugate value ``F* = v^T x* - F(x*)``.

        Raises:
            RuntimeError: If :meth:`preprocess` has not been called.

        """
        if self._problem is None or self._backend is None:
            raise RuntimeError("call preprocess(problem) before solve()")

        cfg = self.config
        c = self._problem.constraint
        pert = self._problem.perturbation
        beta, gamma = theta
        v = self._problem.utility(beta)
        b_use = c.b if b is None else np.asarray(b, dtype=float).ravel()

        if lam0 is not None:
            lam = np.asarray(lam0, dtype=float).copy()
        elif cfg.warm_start and self._lam is not None:
            lam = self._lam.copy()
        else:
            lam = np.zeros(c.num_constraints, dtype=float)

        history: list[float] = []
        status = STATUS_MAX_ITER
        nit = 0
        x_hat = np.zeros(c.num_coords)
        ls_steps: list[float] = []

        for it in range(cfg.max_iter):
            nit = it + 1
            _, x_hat, interior = self._recover(v, lam, gamma)
            r = c.matvec(x_hat) - b_use
            r_inf = float(np.max(np.abs(r))) if r.size else 0.0
            history.append(r_inf)
            if r_inf < cfg.tol:
                status = STATUS_CONVERGED
                nit = it
                break

            # Newton weights: D_i / ell on the active set, 0 at saturated coords.
            weight = np.where(interior, pert.inv_hess_weight(x_hat, gamma) / c.ell, 0.0)
            r_two = float(np.linalg.norm(r))
            eps_k = min(cfg.eps0, r_two)
            eps_k = max(eps_k, cfg.eps_floor)

            direction = self._backend.solve(weight, eps_k, -r)
            dderiv = float(r @ direction)  # grad phi . d  (should be < 0)

            phi0 = self._phi(v, lam, b_use, gamma)
            lam, t, _phi_new, ok = armijo_backtracking(
                lambda lp: self._phi(v, lp, b_use, gamma),
                phi0,
                lam,
                direction,
                dderiv,
                c1=cfg.armijo_c1,
                beta=cfg.armijo_beta,
                max_steps=cfg.max_linesearch,
            )
            ls_steps.append(t)
            if not ok:
                status = STATUS_LINESEARCH_FAILED
                break

        # Final recovery at the accepted multipliers.
        _, x_hat, interior = self._recover(v, lam, gamma)
        r = c.matvec(x_hat) - b_use
        r_inf = float(np.max(np.abs(r))) if r.size else 0.0
        if status == STATUS_CONVERGED or r_inf < cfg.tol:
            status = STATUS_CONVERGED

        if cfg.warm_start:
            self._lam = lam.copy()

        f_conj = float(v @ x_hat - c.ell @ pert.h(x_hat, gamma))
        return PURCResult(
            x=x_hat,
            lam=lam,
            success=(status == STATUS_CONVERGED),
            status=status,
            nit=nit,
            residual=r_inf,
            conjugate=f_conj,
            residual_history=history,
            extras={
                "active_set_size": int(np.count_nonzero(interior)),
                "linesearch_steps": ls_steps,
                "backend_method": self._backend._options.get("method"),
            },
        )


SOLVERS["ssn"] = RegularizedSSNSolver
