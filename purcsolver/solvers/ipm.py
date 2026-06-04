r"""
Primal-dual interior-point method (Mehrotra predictor-corrector) for the PURC
forward problem, with crossover to the exact semismooth Newton solver.

Solves ``min_x F(x;gamma) - v^T x`` over ``{A x = b, 0 <= x <= 1}`` with separable
strictly-convex ``F(x) = sum_i ell_i h(x_i)``.  Introducing dual multipliers
``lambda`` (equality), ``z >= 0`` (for ``x >= 0``) and ``w >= 0`` (for ``x <= 1``),
the perturbed KKT system is

    ell h'(x) - v - A^T lambda - z + w = 0          (stationarity)
    A x = b                                          (primal feasibility)
    x_i z_i = mu,   (1 - x_i) w_i = mu               (perturbed complementarity)

Each Newton step eliminates ``z, w`` then ``Delta x`` to a **normal-equation
system in Delta lambda** whose matrix ``A diag(1/Theta) A^T`` (with
``Theta_i = ell_i h''(x_i) + z_i/x_i + w_i/(1-x_i) > 0``) is a *weighted graph
Laplacian* -- exactly the system our :class:`LaplacianBackend` solves.  So each
IPM iteration is ONE Laplacian solve (no per-coordinate root-find), the iterate
stays strictly interior (globally robust from any start), and the Mehrotra
predictor-corrector chooses ``mu`` adaptively with no per-instance tuning.

Once ``mu`` is small the iterate is near the (possibly saturated) solution; we
**cross over** to the exact box semismooth Newton solver warm-started at
``lambda``, which identifies the sparse active set (recovering the forest fast
path), removes the barrier bias, and finishes with the quadratic local rate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import SOLVERS
from .base import ForwardSolver
from .ssn import RegularizedSSNSolver
from ..backends.routing import LaplacianBackend
from ..config import SSNConfig
from ..problem import PUMProblem
from ..result import STATUS_CONVERGED, STATUS_MAX_ITER, PURCResult
from ..utils.logging import get_logger
from ..utils.torch_compat import DEFAULT_DTYPE, as_tensor
from ..utils.typing import ArrayLike

_logger = get_logger(__name__)

_FRAC_TO_BOUNDARY = 0.995  # step-length safety factor (fraction-to-boundary rule)
_NE_EPS = 1e-12  # tiny regularizer for the (nullspace-singular) Laplacian normal eqns


class IPMSolver(ForwardSolver):
    """
    Mehrotra predictor-corrector primal-dual IPM with semismooth-Newton crossover.

    Args:
        config: Solver configuration (``tol``, ``max_iter`` for the IPM phase, and
            the LaplacianSolve options forwarded to the backend).
        cross_tol: Switch to the SSN crossover once the IPM complementarity ``mu``
            and the primal/dual residuals fall below this.
        crossover: If ``True``, polish with the exact box SSN warm-started at the
            IPM multipliers; otherwise return the (slightly barrier-biased) IPM
            iterate.
    """

    def __init__(
        self,
        config: Optional[SSNConfig] = None,
        *,
        cross_tol: float = 1e-6,
        crossover: bool = True,
    ) -> None:
        super().__init__(config)
        self._backend: Optional[LaplacianBackend] = None
        self.cross_tol = cross_tol
        self.crossover = crossover
        self._ssn: Optional[RegularizedSSNSolver] = None

    def preprocess(self, problem: PUMProblem) -> None:
        """
        Build the persistent Laplacian backend (and the crossover SSN).

        Args:
            problem: The forward problem whose constraint structure is fixed.

        Raises:
            ValueError: If the perturbation is essentially smooth at a box face
                (``h'`` diverges there, e.g. Shannon/logit entropy).  The primal
                interior-point method steps ``x`` in the primal and so requires
                ``h'`` finite on the closed box; such kernels must be solved in
                the dual.  This is a *provable precondition*, checked once here,
                rather than a runtime failure caught mid-iteration.

        """
        c = problem.constraint
        if not problem.perturbation.admits_primal_interior(c.lo, c.hi):
            raise ValueError(
                f"{type(problem.perturbation).__name__} is essentially smooth at a "
                "box face (h' diverges there), so the primal interior-point method "
                "is ill posed.  Solve this kernel in the dual with "
                "RegularizedSSNSolver (or use the barrier-continuation solver, "
                "which recovers the primal as an interior root and is robust here)."
            )
        self._problem = problem
        self._backend = LaplacianBackend(problem.constraint, self.config.laplacian)
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
        Solve the forward problem at ``theta = (beta, gamma)`` by the IPM.

        Args:
            theta: ``(beta, gamma)``.
            b: Optional equality right-hand side override, shape ``(k,)``.
            lam0: Ignored by the IPM (it uses its own interior start); accepted
                for interface compatibility.

        Returns:
            The forward-solve result (after optional SSN crossover).

        Raises:
            RuntimeError: If :meth:`preprocess` has not been called.

        """
        if self._problem is None or self._backend is None:
            raise RuntimeError("call preprocess(problem) before solve()")
        del lam0  # the IPM chooses its own strictly-interior start

        cfg = self.config
        c = self._problem.constraint
        pert = self._problem.perturbation
        beta, gamma = theta
        v = self._problem.utility(beta)
        b_use = c.b if b is None else as_tensor(b).reshape(-1)
        ell = c.ell
        m = c.num_coords

        # Strictly-interior start: x at the box centre, z, w = 1, lambda = 0.
        x = torch.full((m,), 0.5, dtype=DEFAULT_DTYPE)
        z = torch.ones(m, dtype=DEFAULT_DTYPE)
        w = torch.ones(m, dtype=DEFAULT_DTYPE)
        lam = torch.zeros(c.num_constraints, dtype=DEFAULT_DTYPE)

        history: list[float] = []
        status = STATUS_MAX_ITER
        nit = 0
        with torch.no_grad():
            for it in range(cfg.max_iter):
                nit = it + 1
                hp = pert.hprime(x, gamma)
                hpp = pert.hsecond(x, gamma)
                r_d = ell * hp - v - c.rmatvec(lam) - z + w
                r_p = c.matvec(x) - b_use
                mu = float((x @ z + (1.0 - x) @ w) / (2 * m))
                feas = max(float(r_d.abs().max()), float(r_p.abs().max()))
                history.append(feas)
                if feas < self.cross_tol and mu < self.cross_tol:
                    status = STATUS_CONVERGED
                    break

                theta_diag = ell * hpp + z / x + w / (1.0 - x)
                inv_theta = 1.0 / theta_diag
                # Factorize A diag(1/Theta) A^T + eps I ONCE; the predictor and
                # corrector reuse it (two triangular solves, one factorization).
                self._backend.prepare(inv_theta, _NE_EPS)

                # --- predictor (affine) step: target mu = 0 ---
                r_z = x * z
                r_w = (1.0 - x) * w
                dx_a, dlam_a, dz_a, dw_a = self._newton_step(
                    c, inv_theta, r_d, r_p, r_z, r_w, x, z, w
                )
                a_p = self._max_step_primal(x, dx_a)
                a_d = self._max_step_dual(z, w, dz_a, dw_a)
                mu_aff = float(
                    ((x + a_p * dx_a) @ (z + a_d * dz_a)
                     + (1.0 - x - a_p * dx_a) @ (w + a_d * dw_a)) / (2 * m)
                )
                sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0

                # --- corrector step: centring + second-order complementarity ---
                r_z = x * z - sigma * mu + dx_a * dz_a
                r_w = (1.0 - x) * w - sigma * mu - dx_a * dw_a
                dx, dlam, dz, dw = self._newton_step(
                    c, inv_theta, r_d, r_p, r_z, r_w, x, z, w
                )
                a_p = _FRAC_TO_BOUNDARY * self._max_step_primal(x, dx)
                a_d = _FRAC_TO_BOUNDARY * self._max_step_dual(z, w, dz, dw)
                x = x + a_p * dx
                lam = lam + a_d * dlam
                z = z + a_d * dz
                w = w + a_d * dw

        ipm_nit = nit
        if self.crossover and self._ssn is not None:
            # Polish to the exact (mu=0) box solution; warm-start at the IPM dual.
            res = self._ssn.solve(theta, b=b, lam0=lam)
            res.nit = ipm_nit + res.nit
            res.residual_history = history + res.residual_history
            res.extras["ipm_nit"] = ipm_nit
            res.extras["phase"] = "ipm+ssn"
            return res

        # Return the (barrier-biased) IPM iterate directly.
        f_conj = float((v @ x) - (ell @ pert.h(x, gamma)))
        r = c.matvec(x) - b_use
        r_inf = float(r.abs().max())
        return PURCResult(
            x=x,
            lam=lam,
            success=(r_inf < cfg.tol),
            status=status,
            nit=ipm_nit,
            residual=r_inf,
            conjugate=f_conj,
            residual_history=history,
            extras={"phase": "ipm", "mu": mu},
        )

    def _newton_step(self, c, inv_theta, r_d, r_p, r_z, r_w, x, z, w):
        """
        One IPM Newton solve via the Laplacian normal equations.

        Args:
            c: The constraint polytope.
            inv_theta: ``1/Theta`` per coordinate (the arc weights).
            r_d: Stationarity residual.
            r_p: Primal residual.
            r_z: Lower-complementarity residual ``x z - sigma mu (+ corr)``.
            r_w: Upper-complementarity residual ``(1-x) w - sigma mu (+ corr)``.
            x: Current primal.
            z: Current lower dual.
            w: Current upper dual.

        Returns:
            ``(dx, dlam, dz, dw)`` Newton directions.

        """
        rhs_x = -r_d - r_z / x + r_w / (1.0 - x)
        rhs_lam = -r_p - c.matvec(inv_theta * rhs_x)
        dlam = self._backend.solve_prepared(rhs_lam)  # reuse the per-iteration factorization
        dx = inv_theta * (rhs_x + c.rmatvec(dlam))
        dz = -(r_z + z * dx) / x
        dw = (-r_w + w * dx) / (1.0 - x)
        return dx, dlam, dz, dw

    @staticmethod
    def _max_step_primal(x: torch.Tensor, dx: torch.Tensor) -> float:
        """
        Largest ``alpha in (0,1]`` keeping ``x + alpha dx`` inside ``(0, 1)``.

        Args:
            x: Current primal (strictly interior).
            dx: Primal direction.

        Returns:
            The fraction-to-boundary step length.

        """
        alpha = torch.ones((), dtype=DEFAULT_DTYPE)
        neg = dx < 0
        if bool(neg.any()):
            alpha = torch.minimum(alpha, (-x[neg] / dx[neg]).min())
        pos = dx > 0
        if bool(pos.any()):
            alpha = torch.minimum(alpha, ((1.0 - x[pos]) / dx[pos]).min())
        return float(alpha.clamp_min(0.0))

    @staticmethod
    def _max_step_dual(z, w, dz, dw) -> float:
        """
        Largest ``alpha in (0,1]`` keeping ``z, w >= 0``.

        Args:
            z: Lower dual.
            w: Upper dual.
            dz: Lower-dual direction.
            dw: Upper-dual direction.

        Returns:
            The fraction-to-boundary step length.

        """
        alpha = torch.ones((), dtype=DEFAULT_DTYPE)
        for s, ds in ((z, dz), (w, dw)):
            neg = ds < 0
            if bool(neg.any()):
                alpha = torch.minimum(alpha, (-s[neg] / ds[neg]).min())
        return float(alpha.clamp_min(0.0))


SOLVERS["ipm"] = IPMSolver
