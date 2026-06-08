r"""
Trust-region damped-BFGS minimizer for the debiased-FY outer problem (torch float64).

    min Q(theta)  over  {beta in R^K free, gamma in Gamma}

with ``Gamma`` the nonneg box ``{gamma >= 0}`` or the Bernstein polyhedron
``{M gamma >= -1}``.  Design (see CLAUDE.md / the outer-problem-regularity note):

* **gradient-only** -- the curvature model ``B`` is built from secant pairs
  (Powell-damped BFGS, kept PD); no finite differences, no flow sensitivity.
* **no line search** -- an ``ell_inf`` trust region makes the step a small box+linear
  QP (:func:`solve_box_linear_qp`, scipy-free); a ratio test governs the radius.
* **convergence** -- global to a stationary (hence, by convexity, global) point under
  uniform continuity + bounded models; local superlinear (Dennis--Moore) at the PD
  Hessian; the radius stops binding near ``theta*`` so the *original* ``grad Q = 0`` is
  solved (no regularization bias).

The expensive oracle (one batched inner solve per ``value_and_grad``) dominates cost;
this loop is deliberately light and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor
from .qp import solve_box_linear_qp


@dataclass
class TRConfig:
    """Trust-region BFGS configuration."""

    max_iter: int = 200
    tol: float = 1e-7          # projected-gradient sup-norm stop
    delta0: float = 1.0        # initial ell_inf radius
    delta_max: float = 1e3
    delta_min: float = 1e-14   # stall guard
    eta1: float = 0.1          # accept threshold (rho)
    eta2: float = 0.75         # expand threshold
    shrink: float = 0.25
    expand: float = 2.0


@dataclass
class TRResult:
    """Outcome of a trust-region BFGS minimization."""

    theta: torch.Tensor
    objective: float
    grad: torch.Tensor
    n_outer: int
    converged: bool
    gmap: float


class TrustRegionBFGS:
    """Trust-region damped-BFGS minimizer over a box or Bernstein polyhedron."""

    def __init__(self, config: TRConfig | None = None) -> None:
        self.config = config or TRConfig()

    def _project_gamma(self, gamma, bernstein_M):
        """Euclidean projection of a gamma point onto ``Gamma`` (criticality measure)."""
        if bernstein_M is None:
            return torch.clamp(gamma, min=0.0)
        # min 1/2||x - gamma||^2 s.t. M x >= -1  (x = d, feasible start d=0)
        ng = gamma.shape[0]
        big = torch.full((ng,), 1e12, dtype=gamma.dtype)
        a = -torch.ones(bernstein_M.shape[0], dtype=gamma.dtype)
        res = solve_box_linear_qp(torch.eye(ng, dtype=gamma.dtype), -gamma, -big, big,
                                  A=bernstein_M, a=a)
        return res.d

    def _criticality(self, theta, g, n_beta, bernstein_M):
        """``||theta - P_Gamma(theta - g)||_inf`` (beta block is unconstrained)."""
        step = theta - g
        beta = step[:n_beta]
        gamma_p = self._project_gamma(step[n_beta:], bernstein_M)
        proj = torch.cat([beta, gamma_p])
        return float(torch.max(torch.abs(theta - proj)))

    def _subproblem_box(self, theta, n_beta, bernstein_M, delta):
        """Box (lo,hi) and optional linear (A,a) rows for the TR subproblem at ``theta``."""
        P = theta.shape[0]
        lo = torch.full((P,), -delta, dtype=theta.dtype)
        hi = torch.full((P,), delta, dtype=theta.dtype)
        if bernstein_M is None:  # nonneg: gamma + d >= 0 folds into a box lower bound
            gamma = theta[n_beta:]
            lo[n_beta:] = torch.maximum(lo[n_beta:], -gamma)
            return lo, hi, None, None
        # Bernstein: [0 | M] d >= -1 - M gamma
        A = torch.zeros((bernstein_M.shape[0], P), dtype=theta.dtype)
        A[:, n_beta:] = bernstein_M
        a = -1.0 - bernstein_M @ theta[n_beta:]
        return lo, hi, A, a

    def minimize(self, value_and_grad, theta0, n_beta, *, bernstein_M=None) -> TRResult:
        """
        Minimize ``Q`` from ``theta0``.

        Args:
            value_and_grad: ``theta (torch) -> (float Q, torch grad)`` -- the only oracle.
            theta0: Flat start ``[beta, gamma]`` (torch or array).
            n_beta: Number of unconstrained ``beta`` coordinates ``K``.
            bernstein_M: ``None`` for the nonneg box, else the Bernstein matrix ``M``
                so the feasible set is ``{M gamma >= -1}``.

        Returns:
            A :class:`TRResult`.

        """
        cfg = self.config
        theta = as_tensor(theta0).to(DEFAULT_DTYPE).reshape(-1).clone()
        bM = None if bernstein_M is None else as_tensor(bernstein_M).to(DEFAULT_DTYPE)
        P = theta.shape[0]
        Q, g = value_and_grad(theta)
        g = as_tensor(g).to(DEFAULT_DTYPE).reshape(-1)
        B = torch.eye(P, dtype=DEFAULT_DTYPE)
        delta = cfg.delta0
        nit, converged, gmap = 0, False, float("inf")
        for it in range(cfg.max_iter):
            nit = it + 1
            lo, hi, A, a = self._subproblem_box(theta, n_beta, bM, delta)
            d = solve_box_linear_qp(B, g, lo, hi, A=A, a=a).d
            pred = -float(g @ d + 0.5 * d @ (B @ d))
            if float(torch.max(torch.abs(d))) > 1e-14 and pred > 0.0:
                Qt, gt = value_and_grad(theta + d)
                gt = as_tensor(gt).to(DEFAULT_DTYPE).reshape(-1)
                rho = (Q - Qt) / pred
                if rho >= cfg.eta1:  # accept + Powell-damped BFGS update
                    s, y = d, gt - g
                    Bs = B @ s
                    sBs, sy = float(s @ Bs), float(s @ y)
                    if sy < 0.2 * sBs:
                        th = 0.8 * sBs / (sBs - sy)
                        y = th * y + (1.0 - th) * Bs
                        sy = float(s @ y)
                    if sBs > 0 and sy > 0:
                        B = B - torch.outer(Bs, Bs) / sBs + torch.outer(y, y) / sy
                    theta, Q, g = theta + d, Qt, gt
                    if rho > cfg.eta2 and float(torch.max(torch.abs(d))) >= 0.9 * delta:
                        delta = min(cfg.expand * delta, cfg.delta_max)
                else:
                    delta *= cfg.shrink
            else:
                delta *= cfg.shrink
            gmap = self._criticality(theta, g, n_beta, bM)
            if gmap < cfg.tol or delta < cfg.delta_min:
                converged = gmap < cfg.tol
                break
        return TRResult(theta=theta, objective=float(Q), grad=g, n_outer=nit,
                        converged=converged, gmap=gmap)
