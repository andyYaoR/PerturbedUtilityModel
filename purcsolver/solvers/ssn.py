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

Each iteration solves the regularized Newton system ``(H + eps I) d = -r`` (the
linear solve is delegated to LaplacianSolve via :class:`LaplacianBackend`) and
globalizes it with a **Levenberg-Marquardt trust region**: the damping ``eps`` is
adapted from the actual-vs-predicted reduction ratio ``rho`` of the convex dual
``phi``.  This converges from *any* start -- including the empty-active-set
``lambda=0`` where ``H=0`` for hard-saturation perturbations (quadratic, sieve) --
needs no per-instance tuning (``eps`` self-scales), and **preserves the sparse
active set** (no smoothing / barrier).  ``eps`` shrinks toward a pure Newton step
as the model becomes trustworthy, giving the local Q-quadratic rate from strong
semismoothness; it grows only when a step is poor (rank-deficient ``H``).

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
from .base import ForwardSolver

_logger = get_logger(__name__)

# Levenberg-Marquardt / trust-region constants (problem-independent -> no
# per-instance tuning).  rho = actual/predicted reduction of the convex dual
# decides whether to accept the step and how to rescale the damping eps.
_LM_ACCEPT = 0.1  # accept the step if rho exceeds this
_LM_GOOD = 0.75  # rho above this -> model trustworthy -> reduce damping (-> Newton)
_LM_POOR = 0.25  # rho below this (but accepted) -> increase damping
_LM_INC = 4.0  # damping growth factor on a poor / rejected step
_LM_DEC = 0.25  # damping shrink factor on a good step
_LM_EPS_MAX = 1e12  # damping cap
_PHI_NOISE = 1e-12  # relative floor below which dual-objective differences are noise


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
        self._lam_batch: Optional[torch.Tensor] = None
        self._lm_eps: Optional[float] = None  # persisted LM damping (warm-started)
        self._lm_eps_batch: Optional[torch.Tensor] = None  # per-system LM damping

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

    def _phi(self, v: torch.Tensor, lam: torch.Tensor, b: torch.Tensor, gamma: ArrayLike) -> float:
        """
        Evaluate the convex dual objective ``phi(lambda)``.

        Args:
            v: Link utilities.
            lam: Multipliers.
            b: Equality right-hand side.
            gamma: Perturbation parameters.

        Returns:
            ``phi(lambda) = -b^T lambda + sum_i ell_i h*(eta_i)``.

        """
        c = self._problem.constraint
        pert = self._problem.perturbation
        eta = (v + c.rmatvec(lam)) / c.ell
        conj = pert.conj_box(eta, c.lo, c.hi, gamma)
        return float(-(b @ lam) + (c.ell @ conj))

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
        eps_trace: list[float] = []
        # Levenberg-Marquardt damping, warm-started across solves within a problem.
        eps = self._lm_eps if (cfg.warm_start and self._lm_eps is not None) else cfg.lm_eps0

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

                # Newton weights: D_i / ell on the active set, 0 at saturated coords.
                weight = torch.where(
                    interior,
                    pert.inv_hess_weight(x_hat, gamma) / c.ell,
                    torch.zeros_like(x_hat),
                )
                phi0 = float(-(b_use @ lam)) + float(c.ell @ (eta * x_hat - pert.h(x_hat, gamma)))

                # Levenberg-Marquardt trust region: solve (H + eps I) d = -r and
                # adapt eps from the actual-vs-predicted reduction ratio rho.
                pred_floor = _PHI_NOISE * (abs(phi0) + 1.0)  # below this, phi-diffs are noise
                accepted = False
                converged_floor = False
                for _try in range(cfg.lm_max_tries):
                    eps = min(max(eps, cfg.eps_floor), _LM_EPS_MAX)
                    direction = self._backend.solve(weight, eps, -r)
                    rd = float(r @ direction)  # grad phi . d  (< 0)
                    dd = float(direction @ direction)
                    pred = -0.5 * rd + 0.5 * eps * dd  # predicted dual decrease (> 0)
                    if pred <= pred_floor:
                        # Predicted decrease is below phi's numerical precision:
                        # we are at a stationary point of the convex dual = the
                        # global optimum (rho would just be noise).
                        lam = lam + direction
                        accepted = True
                        converged_floor = True
                        break
                    phi_new = self._phi(v, lam + direction, b_use, gamma)
                    rho = (phi0 - phi_new) / pred
                    if rho < _LM_ACCEPT:
                        eps *= _LM_INC  # poor/negative step: shrink the trust region
                        continue
                    lam = lam + direction
                    if rho > _LM_GOOD:
                        eps *= _LM_DEC  # model trustworthy: expand toward Newton
                    elif rho < _LM_POOR:
                        eps *= _LM_INC
                    accepted = True
                    break
                eps_trace.append(eps)
                if converged_floor:
                    status = STATUS_CONVERGED
                    break
                if not accepted:
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
            self._lm_eps = eps

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
                "lm_eps_trace": eps_trace,
                "lm_eps_final": eps,
                "backend_method": self._backend.method,
                "backend_phase": self._backend.phase,
            },
        )

    def solve_batch(
        self,
        theta: Tuple[ArrayLike, ArrayLike],
        b_batch: ArrayLike,
        *,
        lam0: Optional[ArrayLike] = None,
    ) -> PURCResult:
        """
        Solve a batch of OD-pairs sharing ``(perturbation, constraint, theta)``.

        The OD-pairs share the network ``A``, the link utilities ``v(beta)`` and
        the perturbation, and differ only in the demand ``b``.  The whole Newton
        loop is vectorized over the ``B`` systems: recovery, residual, weights and
        the line search operate on ``[B, N]`` / ``[B, k]`` tensors, and each
        Newton iteration is a single batched linear solve.  A per-system converged
        mask freezes finished systems.

        Args:
            theta: ``(beta, gamma)`` shared by all systems.
            b_batch: Per-system demands, shape ``[B, k]``.
            lam0: Optional warm-start multipliers ``[B, k]``; defaults to the
                persisted batch (when shapes match) or zeros.

        Returns:
            A :class:`PURCResult` whose ``x`` (``[B, N]``) and ``lam`` (``[B, k]``)
            are batched; ``success`` is overall, with per-system residuals and the
            converged mask in ``extras``.

        Raises:
            RuntimeError: If :meth:`preprocess` has not been called.

        """
        if self._problem is None or self._backend is None:
            raise RuntimeError("call preprocess(problem) before solve_batch()")

        cfg = self.config
        c = self._problem.constraint
        pert = self._problem.perturbation
        beta, gamma = theta
        v = self._problem.utility(beta).reshape(1, -1)  # [1, N]
        b_batch = as_tensor(b_batch)
        if b_batch.ndim == 1:
            b_batch = b_batch.reshape(1, -1)
        n_sys = b_batch.shape[0]

        if lam0 is not None:
            lam = as_tensor(lam0).clone()
        elif (
            cfg.warm_start
            and self._lam_batch is not None
            and self._lam_batch.shape == b_batch.shape
        ):
            lam = self._lam_batch.clone()
        else:
            lam = torch.zeros_like(b_batch)

        ell = c.ell.reshape(1, -1)
        lo, hi = c.lo, c.hi
        history: list[float] = []
        converged = torch.zeros(n_sys, dtype=torch.bool)
        status = STATUS_MAX_ITER
        nit = 0
        x_hat = torch.zeros((n_sys, c.num_coords), dtype=DEFAULT_DTYPE)
        if (
            cfg.warm_start
            and self._lm_eps_batch is not None
            and self._lm_eps_batch.shape[0] == n_sys
        ):
            eps = self._lm_eps_batch.clone()
        else:
            eps = torch.full((n_sys,), cfg.lm_eps0, dtype=DEFAULT_DTYPE)

        def phi_batch(lvec):
            eta_t = (v + c.rmatvec_batch(lvec)) / ell
            conj = pert.conj_box(eta_t, lo, hi, gamma)
            return -(b_batch * lvec).sum(dim=1) + (ell * conj).sum(dim=1)

        with torch.no_grad():
            for it in range(cfg.max_iter):
                nit = it + 1
                eta = (v + c.rmatvec_batch(lam)) / ell
                x_hat, interior = pert.primal_recovery(eta, lo, hi, gamma)
                r = c.matvec_batch(x_hat) - b_batch
                r_inf = r.abs().amax(dim=1)
                history.append(float(r_inf.max()))
                converged = converged | (r_inf < cfg.tol)
                if bool(converged.all()):
                    status = STATUS_CONVERGED
                    nit = it
                    break

                weight = torch.where(
                    interior, pert.inv_hess_weight(x_hat, gamma) / ell, torch.zeros_like(x_hat)
                )
                phi0 = -(b_batch * lam).sum(dim=1) + (
                    ell * (eta * x_hat - pert.h(x_hat, gamma))
                ).sum(dim=1)
                pred_floor = _PHI_NOISE * (phi0.abs() + 1.0)

                # Per-system Levenberg-Marquardt trust region: each OD-pair adapts
                # its own damping eps from its rho.  Converged systems take no step.
                step_done = converged.clone()
                for _try in range(cfg.lm_max_tries):
                    if bool(step_done.all()):
                        break
                    eps = eps.clamp(cfg.eps_floor, _LM_EPS_MAX)
                    direction = self._backend.solve_batch(weight, eps, -r)
                    rd = (r * direction).sum(dim=1)
                    dd = (direction * direction).sum(dim=1)
                    pred = -0.5 * rd + 0.5 * eps * dd
                    floor_hit = (~step_done) & (pred <= pred_floor)
                    safe_pred = pred.clamp_min(1e-300)
                    rho = (phi0 - phi_batch(lam + direction)) / safe_pred
                    accept = (~step_done) & ((rho >= _LM_ACCEPT) | floor_hit)
                    lam = torch.where(accept.unsqueeze(1), lam + direction, lam)
                    converged = converged | floor_hit
                    grew = (~step_done) & ~accept
                    grew = grew | (accept & (rho < _LM_POOR) & ~floor_hit)
                    shrank = accept & (rho > _LM_GOOD) & ~floor_hit
                    eps = torch.where(grew, eps * _LM_INC, eps)
                    eps = torch.where(shrank, eps * _LM_DEC, eps)
                    step_done = step_done | accept

            eta = (v + c.rmatvec_batch(lam)) / ell
            x_hat, interior = pert.primal_recovery(eta, lo, hi, gamma)
            r = c.matvec_batch(x_hat) - b_batch
            r_inf = r.abs().amax(dim=1)
            converged = converged | (r_inf < cfg.tol)
            if bool(converged.all()):
                status = STATUS_CONVERGED
            f_conj = (v * x_hat).sum(dim=1) - (ell * pert.h(x_hat, gamma)).sum(dim=1)

        if cfg.warm_start:
            self._lam_batch = lam.clone()
            self._lm_eps_batch = eps.clone()

        return PURCResult(
            x=x_hat,
            lam=lam,
            success=bool(converged.all()),
            status=status,
            nit=nit,
            residual=float(r_inf.max()),
            conjugate=f_conj,
            residual_history=history,
            extras={
                "n_systems": n_sys,
                "per_system_residual": r_inf,
                "converged_mask": converged,
                "backend_method": self._backend.method,
                "backend_phase": self._backend.phase,
            },
        )


SOLVERS["ssn"] = RegularizedSSNSolver
