r"""
The debiased (and naive) Fenchel--Young loss for the static PURC model (torch).

For OD pair ``b`` with link weights ``ell``, empirical link frequencies
``ybar_b``, and degree-``l`` U-statistics ``U_l``,

    tilde_ell_b(theta) = sum_ij ell_ij [ 1/2 U_2 + sum_{l=3}^{min(D_b,L)} (gamma_l/l) U_l ]
                         + F*_b(v(beta); gamma) - v(beta)^T ybar_b,

with ``v(beta) = Z beta`` and ``F*_b`` the surplus (``PURCResult.conjugate`` from
the inner solve at ``theta``).  The sample objective is the mean over OD pairs.
Being a Fenchel--Young loss, its gradient is in **residual form** (no
differentiation through ``x*``):

    grad_beta    = (1/B) sum_b Z^T (x*_b(theta) - ybar_b),
    grad_gamma_l = (1/B) sum_{b: D_b>=l} (1/l) sum_ij ell_ij ( U_l(ij,b) - x*_ij,b^l ).

:class:`NaiveFYLoss` is the same with the *biased* plug-in ``U_l -> ybar^l``.
All math is torch (float64, CPU by default), consistent with the solver; the
gradient is computed in closed form rather than via autograd.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from ...static_purc.perturbations._bernstein import is_convex
from ...static_purc.problem import PUMProblem
from ...static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor, to_numpy
from .packing import ParamLayout
from .ustats import u_statistics


class DebiasedFYLoss:
    """
    Debiased Fenchel--Young loss with value, gradient, and per-OD scores (torch).

    Args:
        problem: The PURC forward problem (supplies ``ell``, ``Z``, the demand
            dimension, and the perturbation/conjugate).
        solver: A preprocessed forward solver exposing ``solve_batch`` (the
            batched IPM); used to evaluate ``x*_b(theta)`` and ``F*_b``.
        data: The per-OD link data (counts, frequencies, demands).
        L: Highest sieve degree.
        debias: If ``True`` (default) use falling-factorial U-statistics; if
            ``False`` use the biased plug-in ``ybar^l`` (the naive FY loss).
        warm_start: Carry the inner multipliers across ``theta`` evaluations.

    """

    def __init__(
        self,
        problem: PUMProblem,
        solver,
        data,
        L: int,
        *,
        debias: bool = True,
        warm_start: bool = True,
    ) -> None:
        self.problem = problem
        self.solver = solver
        self.data = data
        self.L = L
        self.debias = debias
        self.warm_start = warm_start
        self.layout = ParamLayout(problem.num_params, L - 2)
        self.ell = as_tensor(problem.constraint.ell).to(DEFAULT_DTYPE).reshape(1, -1)  # [1, N]
        self.ybar = as_tensor(data.ybar).to(DEFAULT_DTYPE)  # [B, N]
        self.b_batch = as_tensor(data.b_batch).to(DEFAULT_DTYPE)  # [B, k]
        self.B = int(self.ybar.shape[0])

        if debias:
            self.U, self.valid = u_statistics(data.n_counts, data.D, L)
        else:
            U = torch.stack([self.ybar**deg for deg in range(L + 1)], dim=2)  # [B, N, L+1]
            self.U = U
            self.valid = torch.ones((self.B, L + 1), dtype=torch.bool)

        Z = problem.Z
        if Z is None:
            self._Z = None
        elif isinstance(Z, torch.Tensor):
            self._Z = Z.to(DEFAULT_DTYPE)
        else:
            arr = Z.toarray() if sp.issparse(Z) else np.asarray(Z)
            self._Z = as_tensor(arr).to(DEFAULT_DTYPE)
        self._warm: Optional[torch.Tensor] = None

    def _solve(self, beta: torch.Tensor, gamma: torch.Tensor):
        """Solve ``x*_b`` and the conjugate for all OD pairs at ``(beta, gamma)``."""
        res = self.solver.solve_batch((beta, gamma), self.b_batch, lam0=self._warm)
        if self.warm_start:
            self._warm = res.lam
        return as_tensor(res.x).to(DEFAULT_DTYPE), as_tensor(res.conjugate).to(DEFAULT_DTYPE)

    def value_and_grad(self, theta) -> Tuple[float, torch.Tensor]:
        """
        Objective ``Q_B(theta)`` and its gradient (residual form).

        Args:
            theta: Flat parameter tensor ``[beta, gamma]``.

        Returns:
            ``(Q, grad)`` with ``grad`` a flat tensor like ``theta``.

        """
        beta, gamma = self.layout.unpack(theta)
        # Q_B is finite for every gamma (the surplus is a max over a compact
        # polytope); Gamma_B is a *constraint*, not the domain of an extended-value
        # objective.  But the primal interior-point forward solver requires a
        # strictly convex sieve (h'' > 0, i.e. gamma in Gamma_B) for its normal
        # equations to be positive definite, so for gamma outside Gamma_B we return a
        # non-finite *sentinel* (not the true value) instead of attempting an
        # ill-posed solve.  The optimizer only evaluates feasible iterates; this is
        # reached only by a finite-difference probe straddling the Gamma_B boundary,
        # which the Hessian then differences one-sidedly.
        if gamma.numel() > 0 and not is_convex(to_numpy(gamma)):
            nan = torch.full((self.layout.size,), float("nan"), dtype=DEFAULT_DTYPE)
            return float("inf"), nan
        xstar, fstar = self._solve(beta, gamma)  # [B, N], [B]
        v = self.problem.utility(beta).to(DEFAULT_DTYPE).reshape(-1)  # [N]

        prim = torch.zeros(self.B, dtype=DEFAULT_DTYPE)
        if self.L >= 2:
            ew2 = (self.ell * self.U[:, :, 2]).sum(dim=1)
            prim = prim + torch.where(self.valid[:, 2], 0.5 * ew2, torch.zeros_like(ew2))
        for deg in range(3, self.L + 1):
            gi = deg - 3
            ewl = (self.ell * self.U[:, :, deg]).sum(dim=1)
            term = (gamma[gi] / deg) * ewl
            prim = prim + torch.where(self.valid[:, deg], term, torch.zeros_like(term))
        lin = self.ybar @ v  # [B]
        Q = float(torch.mean(prim + fstar - lin))

        grad = self._scores(xstar, gamma).mean(dim=0)
        return Q, grad

    def value(self, theta) -> float:
        """
        Objective ``Q_B(theta)`` only (for line searches).

        Args:
            theta: Flat parameter tensor.

        Returns:
            The objective value.

        """
        return self.value_and_grad(theta)[0]

    def _scores(self, xstar: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Per-OD score matrix ``[B, P]`` (gradient summands, before averaging).

        Args:
            xstar: Predicted flows ``[B, N]`` at the current ``theta``.
            gamma: Current sieve coefficients ``[L-2]``.

        Returns:
            ``[B, P]`` with column blocks ``[beta (K), gamma (L-2)]``.

        """
        resid = xstar - self.ybar  # [B, N]
        s_beta = resid if self._Z is None else resid @ self._Z  # [B, K]
        s_gamma = torch.zeros((self.B, self.L - 2), dtype=DEFAULT_DTYPE)
        for deg in range(3, self.L + 1):
            gi = deg - 3
            diff = self.U[:, :, deg] - xstar**deg  # [B, N]
            contrib = (self.ell * diff).sum(dim=1) / deg  # [B]
            s_gamma[:, gi] = torch.where(self.valid[:, deg], contrib, torch.zeros_like(contrib))
        return torch.cat([s_beta, s_gamma], dim=1)

    def per_od_scores(self, theta) -> torch.Tensor:
        """
        Per-OD scores ``s_b = grad tilde_ell_b(theta)``, shape ``[B, P]``.

        Args:
            theta: Flat parameter tensor.

        Returns:
            ``[B, P]`` per-OD score matrix (mean over ``b`` is the gradient).

        """
        beta, gamma = self.layout.unpack(theta)
        xstar, _ = self._solve(beta, gamma)
        return self._scores(xstar, gamma)


class NaiveFYLoss(DebiasedFYLoss):
    """
    The (biased) plug-in Fenchel--Young loss: ``U_l`` replaced by ``ybar^l``.

    Provided for the bias contrast in the simulation study (its ``gamma`` score is
    biased away from zero at ``theta_0``, unlike the debiased loss).
    """

    def __init__(self, problem, solver, data, L, *, warm_start: bool = True) -> None:
        super().__init__(problem, solver, data, L, debias=False, warm_start=warm_start)
