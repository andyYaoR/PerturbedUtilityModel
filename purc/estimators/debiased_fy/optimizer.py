"""
Projected damped-Newton estimator for the debiased Fenchel--Young loss (torch).

Minimizes the sample objective ``Q_B`` over ``R^K x Gamma``.  Because ``Q_B`` is
jointly convex but typically *flat* (small FY curvature) and the parameter
dimension ``P = K + (L-2)`` is small, a projected damped-Newton step -- direction
``-H^{-1} g`` from the (ridge-stabilized) finite-difference Hessian ``H``, with a
backtracking Armijo line search and the sieve block projected onto ``Gamma`` --
converges in a handful of iterations where first-order projected gradient crawls.
``beta`` is unconstrained; ``gamma`` is projected before *every* objective
evaluation (a ``gamma`` outside ``Gamma`` makes ``h`` non-convex).  Inner forward
solves use the batched IPM with warm starts carried across steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from ...static_purc.problem import PUMProblem
from ...static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor
from ..base import EstimationResult, Estimator, SimulatedData
from .basis import SieveBasis
from .loss import DebiasedFYLoss
from .projection import GammaProjection
from .tr_optimizer import TRConfig, TrustRegionBFGS
from .variance import hessian_fd


@dataclass
class EstimatorConfig:
    """
    Configuration for the debiased-FY estimator's outer solver.

    The default solver is trust-region BFGS (``method="tr_bfgs"``); the projected
    damped-Newton path (``method="newton"``) is retained for comparison.

    Attributes:
        max_iter: Maximum outer iterations.
        tol_grad: Stop when the projected gradient-mapping sup-norm is below this.
        tol_step: Stop when the parameter step sup-norm is below this.
        tol_obj: Stop when the relative objective decrease falls below this
            (reaches a stationary point even when an active bound makes the
            projected curvature one-sided and the gradient mapping plateaus).
        backtrack: Step-shrink factor in the line search.
        armijo: Armijo sufficient-decrease constant.
        max_ls: Maximum backtracking trials per outer step.
        ridge: Ridge added to the Hessian for positive definiteness.
        fd_step: Finite-difference step for the Hessian.
        proj: The sieve projection (``GammaProjection``).
        method: Outer solver -- ``"newton"`` (projected damped Newton, FD Hessian) or
            ``"tr_bfgs"`` (trust-region BFGS: gradient-only, no FD, no line search,
            handles the box and the Bernstein polyhedron).
        verbose: Print per-iteration diagnostics.

    """

    max_iter: int = 50
    tol_grad: float = 1e-7
    tol_step: float = 1e-10
    tol_obj: float = 1e-12  # stop when the relative objective decrease falls below this
    backtrack: float = 0.5
    armijo: float = 1e-4
    max_ls: int = 40
    ridge: float = 1e-8
    fd_step: float = 1e-5
    proj: GammaProjection = field(default_factory=lambda: GammaProjection("bernstein"))
    basis: object = None  # SieveBasis; None -> monomial (power-series) parametrization
    method: str = "tr_bfgs"  # "tr_bfgs" (gradient-only trust region, default) or "newton"
    verbose: bool = False


class DebiasedFYEstimator(Estimator):
    """
    The debiased Fenchel--Young estimator (projected damped Newton on ``Q_B``).

    Args:
        problem: The PURC forward problem.
        solver: A preprocessed forward solver exposing ``solve_batch`` (the batched
            IPM, ``crossover=False`` for the pure paper algorithm).
        L: Highest sieve degree.
        config: Optimizer configuration.
        theta_init: Optional warm start for ``theta`` (flat); defaults to zeros.

    """

    def __init__(
        self,
        problem: PUMProblem,
        solver,
        L: int,
        config: Optional[EstimatorConfig] = None,
        *,
        theta_init=None,
    ) -> None:
        self.problem = problem
        self.solver = solver
        self.L = L
        self.config = config or EstimatorConfig()
        self.theta_init = theta_init

    def _project(self, theta, layout):
        """Project the gamma block of ``theta`` onto the convexity set (basis-aware)."""
        beta, gamma = layout.unpack(theta)
        return layout.pack(beta, self._proj(gamma))

    def fit(self, data: SimulatedData) -> EstimationResult:
        """
        Fit by projected damped Newton on ``Q_B``.

        Args:
            data: The per-OD link data.

        Returns:
            The estimation result (``theta_hat`` and diagnostics).

        """
        cfg = self.config
        basis = cfg.basis or SieveBasis.monomial(self.L)
        # A fresh basis-aware projection (don't mutate the user's config.proj).
        self._proj = GammaProjection(cfg.proj.kind, basis=basis)
        loss = DebiasedFYLoss(self.problem, self.solver, data, self.L, warm_start=True, basis=basis)
        layout = loss.layout
        P = layout.size
        if self.theta_init is None:
            theta = torch.zeros(P, dtype=DEFAULT_DTYPE)
        else:
            theta = as_tensor(self.theta_init).to(DEFAULT_DTYPE).reshape(-1)
        theta = self._project(theta, layout)
        if cfg.method == "tr_bfgs":
            return self._fit_tr_bfgs(loss, layout, basis, theta)
        eye = torch.eye(P, dtype=DEFAULT_DTYPE)

        Q, g = loss.value_and_grad(theta)
        nit = 0
        converged = False
        ginf_hist: list[float] = []

        def line_search(direction):
            """Projected-arc Armijo along ``-direction``; returns (cand, accepted)."""
            t = 1.0
            for _ in range(cfg.max_ls):
                cand = self._project(theta - t * direction, layout)
                # Decrease measured against the *realized* (post-projection) step,
                # so an active gamma bound is handled correctly.
                realized = float(g @ (cand - theta))
                if realized < 0 and loss.value(cand) <= Q + cfg.armijo * realized:
                    return cand, True
                t *= cfg.backtrack
            return theta, False

        for it in range(cfg.max_iter):
            nit = it + 1
            H = hessian_fd(loss, theta, h=cfg.fd_step) + cfg.ridge * eye
            # Two-metric projection (Bertsekas): a reduced Newton step on the *free*
            # coordinates and a gradient step on coordinates pinned at an active
            # bound.  Binding coordinates are identified generically through the
            # projection itself -- those a small antigradient probe gets clamped on --
            # so this works for the box and the Bernstein polyhedron alike, and is a
            # no-op (full Newton) when nothing binds.
            probe = self._project(theta - 1e-8 * g, layout)
            free = ((theta - 1e-8 * g) - probe).abs() <= 1e-12
            delta = g.clone()  # gradient metric on binding coordinates
            if bool(free.any()):
                idx = torch.where(free)[0]
                Hff = H.index_select(0, idx).index_select(1, idx)
                try:
                    delta[idx] = torch.linalg.solve(Hff, g[idx])
                except RuntimeError:
                    delta[idx] = g[idx]
            if float(g @ delta) <= 0:  # ensure -delta is a descent direction
                delta = g
            theta_new, accepted = line_search(delta)
            if not accepted:
                # Projection may turn the coupled Newton step into an ascent move;
                # the projected gradient is always a descent direction for convex Q.
                theta_new, accepted = line_search(g)
            step_norm = float((theta_new - theta).abs().max())
            q_prev, theta_prev, g_prev = Q, theta, g
            theta = theta_new
            Q, g = loss.value_and_grad(theta)
            if not (Q < float("inf")):
                # The accepted step landed in the non-computable region (Q = +inf): the
                # inner solve there is ill posed, and a warm-start boundary
                # inconsistency let the line search's value() succeed while this
                # re-evaluation fails.  Revert to the last computable iterate and stop
                # with an honest non-convergence -- NOT a stall at the optimum (the
                # default trust-region solver backtracks around such regions; the
                # FD-Hessian Newton path simply declines them).
                theta, Q, g, converged = theta_prev, q_prev, g_prev, False
                break
            # Projected gradient mapping (handles active gamma bounds).
            gmap = theta - self._project(theta - g, layout)
            ginf = float(gmap.abs().max())
            ginf_hist.append(ginf)
            # Objective decrease this step (>= 0 since the line search is monotone).
            obj_dec = q_prev - Q
            if cfg.verbose:
                print(
                    f"  [newton {nit:3d}] Q={Q:.6e} |Gmap|inf={ginf:.2e} step={step_norm:.2e} dQ={obj_dec:.2e}"
                )
            stalled = obj_dec <= cfg.tol_obj * (abs(q_prev) + 1.0)
            if ginf < cfg.tol_grad or step_norm < cfg.tol_step or not accepted or stalled:
                # Stationary to numerical precision: a clean gradient-mapping stop,
                # or the objective can no longer be decreased (e.g. the optimum is on
                # an active gamma bound, where the projected curvature is one-sided).
                converged = (
                    ginf < cfg.tol_grad or stalled or (accepted and step_norm < cfg.tol_step)
                )
                break

        beta, c = layout.unpack(theta)
        gamma = basis.to_monomial(c)  # report gamma in the monomial basis (gamma = T c)
        return EstimationResult(
            theta_hat=theta,  # gamma block is in the basis coordinates c
            beta_hat=beta,
            gamma_hat=gamma,
            objective=Q,
            grad=g,
            n_outer=nit,
            converged=converged,
            extras={"gmap_history": ginf_hist, "c_hat": c, "basis": basis.name},
        )

    def _fit_tr_bfgs(self, loss, layout, basis, theta0) -> EstimationResult:
        """
        Fit by trust-region BFGS: gradient-only (no FD Hessian), line-search free.

        The convexity constraint is handled inside the trust-region QP subproblem:
        ``nonneg`` is the box ``c >= 0`` (monomial only), ``bernstein`` is the
        polyhedron ``(M T) c >= -1`` (the Bernstein matrix pulled back into the basis
        ``c``).  Returns an :class:`EstimationResult` matching the Newton path
        (``theta_hat`` gamma block in the basis ``c``; ``gamma_hat`` in monomials).
        """
        cfg = self.config
        n_beta = layout.size - (self.L - 2)
        if cfg.proj.kind == "nonneg":
            bM = None
        else:
            from ...static_purc.perturbations._bernstein import bernstein_matrix

            bM = as_tensor(basis.constraint_matrix(bernstein_matrix(self.L - 2))).to(DEFAULT_DTYPE)
        tr = TrustRegionBFGS(TRConfig(max_iter=cfg.max_iter, tol=cfg.tol_grad))
        res = tr.minimize(loss.value_and_grad, theta0, n_beta, bernstein_M=bM)
        beta, c = layout.unpack(res.theta)
        gamma = basis.to_monomial(c)
        return EstimationResult(
            theta_hat=res.theta,
            beta_hat=beta,
            gamma_hat=gamma,
            objective=res.objective,
            grad=res.grad,
            n_outer=res.n_outer,
            converged=res.converged,
            extras={"gmap": res.gmap, "c_hat": c, "basis": basis.name, "method": "tr_bfgs"},
        )
