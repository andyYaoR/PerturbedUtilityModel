"""
Sandwich (Godambe) variance for the debiased Fenchel--Young estimator (torch).

The asymptotic variance is ``Var(theta_hat) ~ B^{-1} A^{-1} K A^{-1}`` with

    A = (1/B) sum_b Hess tilde_ell_b(theta_hat)   (the average Hessian),
    K = (1/B) sum_b s_b s_b^T                      (the empirical score covariance),

where ``s_b = grad tilde_ell_b``.  The bread ``A`` is the Jacobian of the
*analytic* gradient ``grad Q_B`` and is obtained by central finite differences
(the forward solver intentionally does not expose ``dx*/dv``, so a closed-form
Hessian would require sensitivities -- the FD route is robust and cheap because
each gradient is one warm-started batched solve).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor


@dataclass
class SandwichResult:
    """
    Sandwich-variance outputs.

    Attributes:
        var: Estimated covariance of ``theta_hat``, shape ``[P, P]``.
        se: Standard errors ``sqrt(diag(var))``, shape ``[P]``.
        A: The bread ``A`` (average Hessian), shape ``[P, P]``.
        K: The meat ``K`` (score covariance), shape ``[P, P]``.
        cond_A: Condition number of ``A`` (flags weak identification).

    """

    var: torch.Tensor
    se: torch.Tensor
    A: torch.Tensor
    K: torch.Tensor
    cond_A: float


def per_od_scores(loss, theta) -> torch.Tensor:
    """
    Per-OD score matrix ``[B, P]`` at ``theta`` (delegates to the loss).

    Args:
        loss: A debiased FY loss.
        theta: Flat parameter tensor.

    Returns:
        ``[B, P]`` per-OD scores.

    """
    return loss.per_od_scores(theta)


def hessian_fd(loss, theta, h: float = 1e-5) -> torch.Tensor:
    """
    Central finite-difference Hessian of ``Q_B`` (Jacobian of the gradient).

    Uses a central difference where both perturbed points lie in the forward
    solver's validity domain (the convexity set ``Gamma_B``, ``h'' > 0``), and falls
    back to a **one-sided** difference at its boundary: a probe just outside
    ``Gamma_B`` makes the inner solve ill posed, so the loss returns a non-finite
    sentinel there and the difference is taken on the feasible side only.  (This is a
    solver-domain matter, not a property of ``Q_B``, which is finite everywhere.)
    Active bounds are handled by the optimizer's two-metric reduction, not here.

    Args:
        loss: A debiased FY loss exposing ``value_and_grad``.
        theta: Flat parameter tensor ``[P]``.
        h: Finite-difference step.

    Returns:
        The symmetrized Hessian ``A``, shape ``[P, P]``.

    """
    theta = theta.to(DEFAULT_DTYPE).reshape(-1)
    P = theta.numel()
    _, g0 = loss.value_and_grad(theta)
    A = torch.zeros((P, P), dtype=DEFAULT_DTYPE)
    for j in range(P):
        e = torch.zeros(P, dtype=DEFAULT_DTYPE)
        e[j] = h
        _, gp = loss.value_and_grad(theta + e)
        _, gm = loss.value_and_grad(theta - e)
        fp = bool(torch.isfinite(gp).all())
        fm = bool(torch.isfinite(gm).all())
        if fp and fm:
            A[:, j] = (gp - gm) / (2.0 * h)
        elif fp:  # theta - e is off-domain: forward difference
            A[:, j] = (gp - g0) / h
        elif fm:  # theta + e is off-domain: backward difference
            A[:, j] = (g0 - gm) / h
        else:  # both off-domain (degenerate): identity column
            A[j, j] = 1.0
    return 0.5 * (A + A.T)


def sandwich_variance(loss, theta_hat, *, h: float = 1e-5) -> SandwichResult:
    """
    Sandwich variance ``B^{-1} A^{-1} K A^{-1}`` at ``theta_hat``.

    Args:
        loss: A debiased FY loss.
        theta_hat: The fitted flat parameter tensor.
        h: Finite-difference step for the Hessian.

    Returns:
        A :class:`SandwichResult` (covariance, standard errors, ``A``, ``K``,
        ``cond(A)``).

    """
    theta_hat = theta_hat.to(DEFAULT_DTYPE).reshape(-1)
    scores = loss.per_od_scores(theta_hat)  # [B, P]
    B = scores.shape[0]
    K = (scores.T @ scores) / B  # [P, P]
    A = hessian_fd(loss, theta_hat, h=h)
    Ainv = torch.linalg.inv(A)
    var = (Ainv @ K @ Ainv) / B
    se = torch.sqrt(torch.clamp(torch.diag(var), min=0.0))
    cond_A = float(torch.linalg.cond(A))
    return SandwichResult(var=var, se=se, A=A, K=K, cond_A=cond_A)


def to_monomial(sw: SandwichResult, basis, n_beta: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Delta-method map of a ``c``-space sandwich result to the monomial basis.

    When the estimator parametrizes the sieve shape in a basis ``c`` with ``gamma =
    T c``, the sandwich ``var``/``se`` are in ``c``-space.  Since the map is linear,
    the monomial covariance is exact: ``J var J^T`` with ``J = blkdiag(I_K, T)``.

    Args:
        sw: A :class:`SandwichResult` in the active (``c``) coordinates.
        basis: The :class:`SieveBasis` used (its ``T`` maps ``c -> gamma``).
        n_beta: Number of utility coefficients ``K`` (the unconstrained block).

    Returns:
        ``(var_monomial, se_monomial)`` for ``theta = [beta, gamma]``.

    """
    P = sw.var.shape[0]
    J = torch.eye(P, dtype=DEFAULT_DTYPE)
    if not basis.is_monomial:
        T = as_tensor(basis.T).to(DEFAULT_DTYPE)
        J[n_beta:, n_beta:] = T
    var_m = J @ sw.var @ J.T
    se_m = torch.sqrt(torch.clamp(torch.diag(var_m), min=0.0))
    return var_m, se_m
