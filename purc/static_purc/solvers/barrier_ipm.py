r"""
Barrier-smoothed dual continuation method for the PURC forward problem.

For a fixed ``mu > 0`` we replace the box constraint by a log-barrier, giving the
smooth, strictly-convex, *essentially-smooth* (on the open box) separable program

    min_x  sum_i [ ell_i h(x_i; gamma) - mu log(x_i - lo_i) - mu log(hi_i - x_i) ]
           - v^T x
    s.t.   A x = b.

Its convex dual ``phi_mu(lambda) = -b^T lambda + sum_i M_mu((v + A^T lambda)_i)``
(with ``M_mu`` the per-coordinate barrier-smoothed conjugate) is **smooth and
unconstrained in lambda**.  Its gradient and generalized Hessian are

    grad phi_mu(lambda) = A x_hat_mu - b =: r,
    Hess phi_mu(lambda) = A diag(weight) A^T,   weight_i = dx_hat_i / dy_i > 0,

where ``x_hat_mu`` is the (always strictly interior) maximiser of the inner
problem -- the root of the strictly-monotone FOC

    ell_i h'(x; gamma) - y_i - mu/(x - lo_i) + mu/(hi_i - x) = 0,   y = v + A^T lambda,

and ``weight = 1 / g_mu''`` (see :func:`recover_barrier_primal`).  So each Newton
step is ONE weighted-graph-Laplacian solve -- exactly our
:class:`LaplacianBackend` -- and, crucially, **there is no active set**: with
``mu > 0`` every coordinate is interior, so the Hessian is the *full* weighted
Laplacian with no combinatorial face selection and hence no active-set thrashing.
As ``mu -> 0`` the weights vanish continuously on saturating arcs, recovering the
sparse solution.

This is robust for **every** strictly-convex kernel, including Legendre-type ones
(Shannon / logit entropy) whose ``h'`` diverges at a box face: the primal is
recovered as an *interior* root of a monotone equation, so ``h'`` is never
evaluated at the singular bound (contrast the primal interior-point method in
:mod:`purc.static_purc.solvers.ipm`, which steps ``x`` in the primal and is therefore
restricted by :meth:`SeparablePerturbation.admits_primal_interior`).

A short geometric continuation ``mu <- mu_factor * mu`` (warm-started across
``mu``) drives ``mu`` down; once it is below ``mu_min`` we **cross over** to the
exact box semismooth Newton solver warm-started at ``lambda`` to remove the
barrier bias and finish at the quadratic local rate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..backends.routing import LaplacianBackend
from ..config import ForwardSolverConfig
from ..problem import PUMProblem
from ..result import STATUS_CONVERGED, STATUS_MAX_ITER, PURCResult
from ..utils.logging import get_logger
from ..utils.torch_compat import DEFAULT_DTYPE, as_tensor
from ..utils.typing import ArrayLike
from . import SOLVERS
from .barrier import BarrierRecoveryConfig, barrier_dual_objective, recover_barrier_primal
from .base import ForwardSolver
from .ssn import RegularizedSSNSolver

_logger = get_logger(__name__)

_NE_EPS = 1e-12  # tiny regularizer for the (nullspace-singular) Laplacian normal eqns
_LS_C1 = 1e-4  # Armijo sufficient-decrease constant for the dual line search
_LS_BETA = 0.5  # Armijo backtracking shrink factor
_LS_MAX = 30  # maximum backtracking steps per Newton iteration


class BarrierContinuationSolver(ForwardSolver):
    """
    Barrier-smoothed dual continuation solver with semismooth-Newton crossover.

    Robust on every strictly-convex separable kernel (the primal is recovered as
    an interior root, so kernels singular at a box face -- Shannon / logit
    entropy -- are handled with no special casing), with each Newton step a
    single Laplacian solve and no active-set combinatorics.

    Args:
        config: Solver configuration (``tol``, ``max_iter`` used as the *total*
            Newton budget across the continuation, plus the forwarded
            LaplacianSolve options).
        mu_min: Terminal barrier parameter; below this we cross over to the exact
            box SSN.
        mu_factor: Geometric continuation factor ``mu <- mu_factor * mu`` in
            ``(0, 1)``.
        inner_max: Maximum Newton steps per fixed ``mu``.
        inner_tol_factor: Inner Newton stops once ``||r||_inf <= max(
            inner_tol_factor * mu, config.tol)`` (loose early, tight near the end).
        crossover: If ``True``, polish with the exact box SSN warm-started at the
            barrier multipliers.
        recovery_config: Controls for the per-coordinate barrier root-find.

    """

    def __init__(
        self,
        config: Optional[ForwardSolverConfig] = None,
        *,
        mu_min: float = 1e-8,
        mu_factor: float = 0.1,
        inner_max: int = 20,
        inner_tol_factor: float = 0.1,
        crossover: bool = True,
        recovery_config: Optional[BarrierRecoveryConfig] = None,
    ) -> None:
        super().__init__(config)
        if not (0.0 < mu_factor < 1.0):
            raise ValueError(f"mu_factor must be in (0, 1), got {mu_factor}")
        if mu_min <= 0.0:
            raise ValueError(f"mu_min must be > 0, got {mu_min}")
        self._backend: Optional[LaplacianBackend] = None
        self._ssn: Optional[RegularizedSSNSolver] = None
        self.mu_min = mu_min
        self.mu_factor = mu_factor
        self.inner_max = inner_max
        self.inner_tol_factor = inner_tol_factor
        self.crossover = crossover
        self.recovery_config = recovery_config or BarrierRecoveryConfig()

    def preprocess(self, problem: PUMProblem) -> None:
        """
        Build the persistent Laplacian backend (and the crossover SSN).

        Args:
            problem: The forward problem whose constraint structure is fixed.

        Raises:
            ValueError: If the box is not finite (the log-barrier needs finite
                ``lo`` and ``hi``).

        """
        c = problem.constraint
        if bool((~torch.isfinite(as_tensor(c.lo))).any() or (~torch.isfinite(as_tensor(c.hi))).any()):
            raise ValueError(
                "barrier continuation requires a finite box [lo, hi]; this "
                "constraint has an unbounded side."
            )
        self._problem = problem
        self._backend = LaplacianBackend(c, self.config.laplacian)
        if self.crossover:
            self._ssn = RegularizedSSNSolver(self.config)
            self._ssn.preprocess(problem)

    def solve(
        self,
        theta: Tuple[ArrayLike, ArrayLike],
        *,
        b: Optional[ArrayLike] = None,
        lam0: Optional[ArrayLike] = None,
    ) -> PURCResult:
        """
        Solve the forward problem at ``theta = (beta, gamma)`` by barrier continuation.

        Args:
            theta: ``(beta, gamma)``.
            b: Optional equality right-hand side override, shape ``(k,)``.
            lam0: Optional warm-start multipliers; defaults to zeros.

        Returns:
            The forward-solve result (after optional SSN crossover).

        Raises:
            RuntimeError: If :meth:`preprocess` has not been called.

        """
        if self._problem is None or self._backend is None:
            raise RuntimeError("call preprocess(problem) before solve()")

        cfg = self.config
        c = self._problem.constraint
        beta, gamma = theta
        v = self._problem.utility(beta)
        b_use = c.b if b is None else as_tensor(b).reshape(-1)
        lam = (
            torch.zeros(c.num_constraints, dtype=DEFAULT_DTYPE)
            if lam0 is None
            else as_tensor(lam0).reshape(-1).clone()
        )

        # Scale-free start: mu0 comparable to the driving utilities so the first
        # subproblem is barrier-dominated (x_hat ~ box centre, trivial Newton),
        # then geometric continuation drives mu -> 0.  mu0 only affects the
        # iteration count, never the limit, so this needs no per-instance tuning.
        mu = max(float(as_tensor(v).abs().max()), 1.0)

        history: list[float] = []
        x = torch.full((c.num_coords,), 0.5, dtype=DEFAULT_DTYPE)
        total_newton = 0
        with torch.no_grad():
            while True:
                inner_tol = max(self.inner_tol_factor * mu, cfg.tol)
                for _ in range(self.inner_max):
                    if total_newton >= cfg.max_iter:
                        break
                    x, weight = recover_barrier_primal(
                        self._problem, v, lam, gamma, mu, self.recovery_config
                    )
                    r = c.matvec(x) - b_use
                    r_inf = float(r.abs().max())
                    history.append(r_inf)
                    if r_inf <= inner_tol:
                        break
                    dlam = self._backend.solve(weight, _NE_EPS, -r)
                    lam = self._line_search(v, b_use, gamma, mu, lam, dlam, r)
                    total_newton += 1
                if mu <= self.mu_min or total_newton >= cfg.max_iter:
                    break
                mu = max(self.mu_factor * mu, self.mu_min)

        if self.crossover and self._ssn is not None:
            # Polish to the exact (mu = 0) box solution; warm-start at the barrier dual.
            res = self._ssn.solve(theta, b=b, lam0=lam)
            res.nit = total_newton + res.nit
            res.residual_history = history + res.residual_history
            res.extras["barrier_nit"] = total_newton
            res.extras["phase"] = "barrier+ssn"
            return res

        # Return the (barrier-biased) iterate directly.
        pert = self._problem.perturbation
        f_conj = float((v @ x) - (c.ell @ pert.h(x, gamma)))
        r = c.matvec(x) - b_use
        r_inf = float(r.abs().max())
        return PURCResult(
            x=x,
            lam=lam,
            success=(r_inf < cfg.tol),
            status=STATUS_CONVERGED if r_inf < cfg.tol else STATUS_MAX_ITER,
            nit=total_newton,
            residual=r_inf,
            conjugate=f_conj,
            residual_history=history,
            extras={"phase": "barrier", "mu": mu},
        )

    def _line_search(
        self,
        v: torch.Tensor,
        b: torch.Tensor,
        gamma: ArrayLike,
        mu: float,
        lam: torch.Tensor,
        dlam: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """
        Armijo backtracking on the smooth barrier dual ``phi_mu`` along ``dlam``.

        Args:
            v: Link utilities.
            b: Equality right-hand side.
            gamma: Perturbation parameters.
            mu: Current barrier parameter.
            lam: Current multipliers.
            dlam: Newton direction.
            r: Dual gradient ``grad phi_mu = A x_hat - b`` at ``lam``.

        Returns:
            The updated multipliers ``lam + t * dlam``.

        """
        phi0 = barrier_dual_objective(self._problem, v, lam, b, gamma, mu, config=self.recovery_config)
        gderiv = float(r @ dlam)  # directional derivative of phi_mu (< 0 for a Newton descent dir)
        t = 1.0
        if gderiv >= 0.0:
            # Not a descent direction (numerical degeneracy); take a safeguarded
            # damped gradient step instead.
            return lam - (mu * 1e-3) * r
        for _ in range(_LS_MAX):
            lam_new = lam + t * dlam
            phi_new = barrier_dual_objective(
                self._problem, v, lam_new, b, gamma, mu, config=self.recovery_config
            )
            if phi_new <= phi0 + _LS_C1 * t * gderiv:
                return lam_new
            t *= _LS_BETA
        return lam + t * dlam


SOLVERS["barrier"] = BarrierContinuationSolver
