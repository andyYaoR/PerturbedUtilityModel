"""
Lean hand-rolled trust-region BFGS for the debiased-FY outer problem -- the
reference implementation the native kernel will mirror.

    min Q(theta)  over  {beta in R^K free, gamma in Gamma}

with Gamma either the nonneg box {gamma >= 0} or the Bernstein polyhedron
{M gamma >= -1}.  Properties it is built to honour (see CLAUDE.md):

  * gradient-only -- NO finite differences, NO sensitivity dx*/dv.  The curvature
    model B_k is built from secant pairs (Powell-damped BFGS, kept PD + bounded).
  * NO line search -- a ratio test governs the radius.
  * the constraint set lives in a small ell_inf trust-region QP subproblem solved
    on the *quadratic model* (pure linear algebra, zero expensive oracle calls).
  * convergence guarantee (TR global, under uniform continuity + bounded models;
    local superlinear by Dennis-Moore at the verified PD Hessian) -- the QP step is
    >= the projected-Cauchy decrease, which is what the global theory needs.
  * vanishing regularization: the radius stops binding near theta*, so the full
    quasi-Newton step is taken and the *original* stationarity grad Q = 0 is solved
    (no ridge bias).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def _tr_subproblem(g, B, delta, A, b):
    """Min g'd + 1/2 d'Bd  s.t.  ||d||_inf <= delta,  A d >= b   (model only, no oracle)."""
    P = g.size
    cons = [LinearConstraint(A, b, np.inf)] if (A is not None and A.shape[0]) else []
    res = minimize(
        lambda d: float(g @ d + 0.5 * d @ (B @ d)), np.zeros(P),
        jac=lambda d: g + B @ d, method="SLSQP",
        bounds=Bounds(-delta * np.ones(P), delta * np.ones(P)), constraints=cons,
        options={"ftol": 1e-16, "maxiter": 200},
    )
    return res.x


def _gamma_rows(theta, n_beta, bern_M):
    """Linear feasibility rows A d >= b for gamma + d_gamma (gamma-block only)."""
    gamma = theta[n_beta:]
    ng = gamma.size
    if bern_M is None:  # nonneg box: gamma + d >= 0
        return np.hstack([np.zeros((ng, n_beta)), np.eye(ng)]), -gamma
    A = np.hstack([np.zeros((bern_M.shape[0], n_beta)), bern_M])  # M(gamma + d) >= -1
    return A, -1.0 - bern_M @ gamma


def _project(theta, n_beta, bern_M):
    """Euclidean projection of the gamma-block onto Gamma (for the criticality measure)."""
    beta, gamma = theta[:n_beta], theta[n_beta:]
    if bern_M is None:
        return np.concatenate([beta, np.maximum(gamma, 0.0)])
    res = minimize(
        lambda x: float(0.5 * np.sum((x - gamma) ** 2)), gamma, jac=lambda x: x - gamma,
        method="SLSQP", constraints=[LinearConstraint(bern_M, -1.0, np.inf)],
        options={"ftol": 1e-16, "maxiter": 200},
    )
    return np.concatenate([beta, res.x])


def tr_bfgs(value_and_grad, theta0, n_beta, *, bernstein_M=None, tol=1e-7, max_iter=300,
            delta0=1.0, delta_max=1e3, delta_min=1e-14, eta1=0.1, eta2=0.75):
    """
    Trust-region damped-BFGS minimizer.  ``value_and_grad(theta_np) -> (float, grad_np)``
    is the only oracle (one batched inner solve per call).  Returns ``(theta, info)``.
    """
    theta = np.asarray(theta0, float).copy()
    P = theta.size
    Q, g = value_and_grad(theta)
    B = np.eye(P)
    delta = delta0
    nit, converged = 0, False
    gmap = float("inf")
    for it in range(max_iter):
        nit = it + 1
        A, b = _gamma_rows(theta, n_beta, bernstein_M)
        d = _tr_subproblem(g, B, delta, A, b)
        pred = -float(g @ d + 0.5 * d @ (B @ d))
        if np.max(np.abs(d)) > 1e-14 and pred > 0.0:
            Qt, gt = value_and_grad(theta + d)  # the only oracle call this iteration
            rho = (Q - Qt) / pred
            if rho >= eta1:  # accept
                s, y = d.copy(), gt - g
                Bs = B @ s
                sBs, sy = float(s @ Bs), float(s @ y)
                if sy < 0.2 * sBs:  # Powell damping -> keep B PD
                    th = 0.8 * sBs / (sBs - sy)
                    y = th * y + (1.0 - th) * Bs
                    sy = float(s @ y)
                if sBs > 0 and sy > 0:
                    B = B - np.outer(Bs, Bs) / sBs + np.outer(y, y) / sy
                theta, Q, g = theta + d, Qt, gt
                if rho > eta2 and np.max(np.abs(d)) >= 0.9 * delta:
                    delta = min(2.0 * delta, delta_max)
            else:
                delta *= 0.25
        else:
            delta *= 0.25
        gmap = float(np.max(np.abs(theta - _project(theta - g, n_beta, bernstein_M))))
        if gmap < tol or delta < delta_min:
            converged = gmap < tol
            break
    return theta, dict(n_outer=nit, converged=converged, Q=float(Q), gmap=gmap)
