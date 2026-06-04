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

All solver math runs on torch tensors (CPU ``float64`` by default) under
``torch.no_grad`` -- torch is the data container, not an autograd graph (the
Fenchel-Young estimation gradient needs only ``x*`` and ``F*``, no ``dx*/dtheta``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..backends.routing import LaplacianBackend
from ..config import SSNConfig
from ..problem import PUMProblem
from ..result import (
    STATUS_CONVERGED,
    STATUS_LINESEARCH_FAILED,
    STATUS_MAX_ITER,
    STATUS_STALLED,
    PURCResult,
)
from ..utils.logging import get_logger
from ..utils.torch_compat import DEFAULT_DTYPE, as_tensor
from ..utils.typing import ArrayLike
from . import SOLVERS
from ._linesearch import armijo_backtracking_t
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
        self._lam: Optional[torch.Tensor] = None

    def preprocess(self, problem: PUMProblem) -> None:
        """
        Build the persistent LaplacianSolve backend from the fixed pattern.

        Args:
            problem: The forward problem whose constraint structure is fixed.

        """
        self._problem = problem
        self._backend = LaplacianBackend(problem.constraint, self.config.laplacian)
        self._lam = torch.zeros(problem.constraint.num_constraints, dtype=DEFAULT_DTYPE)

    # -- internals ----------------------------------------------------------

    def _recover(
        self, v: torch.Tensor, lam: torch.Tensor, gamma: ArrayLike
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        b_use = c.b if b is None else as_tensor(b).reshape(-1)

        if lam0 is not None:
            lam = as_tensor(lam0).reshape(-1).clone()
        elif cfg.warm_start and self._lam is not None:
            lam = self._lam.clone()
        else:
            lam = torch.zeros(c.num_constraints, dtype=DEFAULT_DTYPE)

        history: list[float] = []
        status = STATUS_MAX_ITER
        nit = 0
        x_hat = torch.zeros(c.num_coords, dtype=DEFAULT_DTYPE)
        interior = torch.zeros(c.num_coords, dtype=torch.bool)
        ls_steps: list[float] = []
        best_r = float("inf")
        stalled = 0

        with torch.no_grad():
            for it in range(cfg.max_iter):
                nit = it + 1
                eta, x_hat, interior = self._recover(v, lam, gamma)
                r = c.matvec(x_hat) - b_use
                r_inf = float(r.abs().max()) if r.numel() else 0.0
                history.append(r_inf)
                if r_inf < cfg.tol:
                    status = STATUS_CONVERGED
                    nit = it
                    break

                # Stagnation: residual no longer improving (hit the numerical
                # floor below the requested tol).  Stop instead of spinning.
                if r_inf < best_r * (1.0 - 1e-3):
                    best_r = r_inf
                    stalled = 0
                else:
                    stalled += 1
                    if stalled >= cfg.stall_patience:
                        status = STATUS_STALLED
                        break

                # Newton weights: D_i / ell on the active set, 0 at saturated coords.
                weight = torch.where(
                    interior,
                    pert.inv_hess_weight(x_hat, gamma) / c.ell,
                    torch.zeros_like(x_hat),
                )
                eps_k = min(cfg.eps0, float(r.norm()))
                eps_k = max(eps_k, cfg.eps_floor)

                direction = self._backend.solve(weight, eps_k, -r)
                dderiv = float(r @ direction)  # grad phi . d  (should be < 0)

                # Line search on phi as a function of step length t.  eta(t) =
                # eta + t * (A^T d / ell), so A^T d is computed once (not per
                # backtracking step), and phi(0) reuses the iterate's x_hat
                # (eta * x_hat - h) instead of recomputing the recovery.
                atd_over_ell = c.rmatvec(direction) / c.ell
                b_dot_d = float(b_use @ direction)
                phi_lin0 = float(-(b_use @ lam))
                conj0 = eta * x_hat - pert.h(x_hat, gamma)
                phi0 = phi_lin0 + float(c.ell @ conj0)

                def phi_of_t(t, _eta=eta, _atde=atd_over_ell, _lin=phi_lin0, _bd=b_dot_d):
                    conj = pert.conj_box(_eta + t * _atde, c.lo, c.hi, gamma)
                    return _lin - t * _bd + float(c.ell @ conj)

                t, _phi_new, ok = armijo_backtracking_t(
                    phi_of_t,
                    phi0,
                    dderiv,
                    c1=cfg.armijo_c1,
                    beta=cfg.armijo_beta,
                    max_steps=cfg.max_linesearch,
                )
                lam = lam + t * direction
                ls_steps.append(t)
                if not ok:
                    status = STATUS_LINESEARCH_FAILED
                    break

            # Final recovery at the accepted multipliers.
            _, x_hat, interior = self._recover(v, lam, gamma)
            r = c.matvec(x_hat) - b_use
            r_inf = float(r.abs().max()) if r.numel() else 0.0
            if status == STATUS_CONVERGED or r_inf < cfg.tol:
                status = STATUS_CONVERGED

            f_conj = float((v @ x_hat) - (c.ell @ pert.h(x_hat, gamma)))

        if cfg.warm_start:
            self._lam = lam.clone()

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
                "active_set_size": int(interior.sum()),
                "linesearch_steps": ls_steps,
                "backend_method": self._backend.method,
                "backend_phase": self._backend.phase,
            },
        )


SOLVERS["ssn"] = RegularizedSSNSolver
