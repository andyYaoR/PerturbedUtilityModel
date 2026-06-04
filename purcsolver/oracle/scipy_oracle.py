r"""
Independent dense reference oracle.

Solves the same convex dual as the SSN solver but with a deliberately separate,
dependency-light implementation: dense ``numpy.linalg.solve`` for the Newton
system (no LaplacianSolve, no CSC assembler, no shared solver code).  This makes
it a useful cross-check -- it cannot co-validate a bug in the backend/assembly
path -- and, crucially, it is the ground truth for the polynomial sieve, which
CVXPY cannot express.

For ``N`` small/medium (test fixtures) the dense factorization is fine.  It uses
only the perturbation's ``primal_recovery`` / ``conj_box`` / ``inv_hess_weight``,
so it works for any separable kernel.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..problem import PUMProblem
from ..utils.torch_compat import to_numpy
from ..utils.typing import ArrayLike


def solve_scipy(
    problem: PUMProblem,
    theta: Tuple[ArrayLike, ArrayLike],
    *,
    tol: float = 1e-12,
    max_iter: int = 300,
    eps0: float = 1e-2,
    eps_floor: float = 1e-13,
) -> np.ndarray:
    """
    Solve the forward problem via a dense regularized dual Newton method.

    Args:
        problem: The forward problem.
        theta: Parameters ``(beta, gamma)``.
        tol: Convergence tolerance on ``||A x_hat - b||_inf``.
        max_iter: Maximum Newton iterations.
        eps0: Initial / maximum regularization.
        eps_floor: Lower bound on the regularizer.

    Returns:
        The optimal primal ``x*``, shape ``(N,)``.

    """
    c = problem.constraint
    pert = problem.perturbation
    beta, gamma = theta
    # Bridge all torch data to numpy: this oracle is a deliberately separate,
    # dense numpy implementation (no torch ops, no LaplacianSolve).
    v = to_numpy(problem.utility(beta))
    A = c.A.toarray()
    b = to_numpy(c.b)
    ell, lo, hi = to_numpy(c.ell), to_numpy(c.lo), to_numpy(c.hi)
    k = A.shape[0]
    lam = np.zeros(k)
    eye = np.eye(k)

    def recover(lvec):
        eta = (v + A.T @ lvec) / ell
        x_hat, interior = pert.primal_recovery(eta, lo, hi, gamma)
        return to_numpy(x_hat), to_numpy(interior)

    def phi(lvec):
        eta = (v + A.T @ lvec) / ell
        return float(-b @ lvec + ell @ to_numpy(pert.conj_box(eta, lo, hi, gamma)))

    for _ in range(max_iter):
        x_hat, interior = recover(lam)
        r = A @ x_hat - b
        if np.max(np.abs(r)) < tol:
            break
        weight = to_numpy(pert.inv_hess_weight(x_hat, gamma)) / ell
        D = np.where(interior, weight, 0.0)
        H = A @ (D[:, None] * A.T)
        eps = max(min(eps0, float(np.linalg.norm(r))), eps_floor)
        d = np.linalg.solve(H + eps * eye, -r)
        phi0 = phi(lam)
        g = float(r @ d)
        t = 1.0
        for _ in range(50):
            if phi(lam + t * d) <= phi0 + 1e-4 * t * g:
                break
            t *= 0.5
        lam = lam + t * d

    x_hat, _ = recover(lam)
    return np.asarray(x_hat, dtype=float)
