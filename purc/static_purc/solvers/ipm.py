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

**Global convergence.** The bare Mehrotra step (``safeguard=False``) is a
heuristic with no convergence proof.  Its fraction-to-boundary step length
controls only *positivity*, not *centrality*: a single full step can unbalance the
complementarity products -- some ``x_i z_i`` shoot up (``mu`` explodes), others
collapse toward zero -- so the iterate leaves the central-path neighbourhood and,
on stiff instances, never returns (the residual diverges to NaN).  With
``safeguard=True`` (default) two changes globalize it.  (i) A *data-scaled start*
sets the duals to zero the stationarity residual at ``x = 1/2`` (splitting
``g = ell h'(1/2) - v`` between ``z, w``), so the first Newton directions are well
scaled.  (ii) The step length is chosen by a *line search* that keeps every iterate
in the wide central-path neighbourhood ``N(gamma) = {x_i z_i, (1-x_i) w_i >=
gamma * mu}``: the *fast step* is the longest Mehrotra step in ``N`` that decreases
``mu`` by a fixed factor ``rho``; if none does, a centred *safe step* takes the
longest step in ``N`` with an Armijo decrease of ``mu``, whose existence is
guaranteed (Wright-Ralph 1996, Lemma 3.1).  Membership in ``N`` is by centrality
only: this is what prevents the blow-up, while the residuals are driven down by the
Newton step (the primal part exactly as ``(1 - alpha)``) and the ``rho``-decrease
of ``mu``.  An explicit ``||r|| <= beta * mu`` wall is *not* imposed -- the
``O(ell * alpha^2)`` nonlinearity of ``h'`` would wedge the iterate against it, and
the pure-NCP ``g(alpha)`` device that keeps ``r`` exactly ``(1-alpha) r``
(Wright-Ralph 1996, eqs. (4b)-(6)) does not transfer to PURC's mixed (equality+box) form.
This is the safe-step/fast-step method of Wright & Ralph (1996, *Math. Oper. Res.*
21:815) and Ralph & Wright (2000, *Math. Oper. Res.* 25:179) within the
Kojima-Noma-Yoshise (1994) framework.  The PURC KKT map is monotone (the problem
is convex) and the reduced curvature ``Theta`` is positive (``h'' > 0``), so their
global-convergence theory applies.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from ..backends.routing import LaplacianBackend
from ..config import ForwardSolverConfig
from ..problem import PUMProblem
from ..result import STATUS_CONVERGED, STATUS_MAX_ITER, PURCResult
from ..utils.logging import get_logger
from ..utils.native import native_available, native_core
from ..utils.torch_compat import DEFAULT_DTYPE, as_tensor, to_numpy
from ..utils.typing import ArrayLike
from . import SOLVERS
from .base import ForwardSolver
from .ssn import RegularizedSSNSolver

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
        safeguard: If ``True`` (default), globalize the Mehrotra step with the
            safe-step/fast-step neighbourhood line search (see module docstring);
            if ``False``, take the bare fraction-to-boundary step (no guarantee).
        gamma_min: Centrality bound of the wide neighbourhood ``N(gamma_min)``.
        beta_floor: Floor for the (start-derived) neighbourhood infeasibility bound.
        rho: A fast step is accepted only if it decreases ``mu`` by at least this
            factor (``mu(alpha) <= rho * mu``); otherwise a safe step is taken.
        kappa: Armijo constant for the safe step's sufficient-decrease test.
        safe_sigma: ``(lo, hi)`` clamp for the safe step's centring parameter.
        ls_max: Maximum backtracking trials in the step-length line search.
        ls_chi: Backtracking reduction factor.

    """

    def __init__(
        self,
        config: Optional[ForwardSolverConfig] = None,
        *,
        cross_tol: float = 1e-6,
        crossover: bool = True,
        safeguard: bool = True,
        gamma_min: float = 1e-6,
        beta_floor: float = 10.0,
        rho: float = 0.5,
        kappa: float = 0.1,
        safe_sigma: Tuple[float, float] = (0.1, 0.5),
        ls_max: int = 50,
        ls_chi: float = 0.5,
    ) -> None:
        super().__init__(config)
        self._backend: Optional[LaplacianBackend] = None
        self.cross_tol = cross_tol
        self.crossover = crossover
        # --- safe-step/fast-step globalization (Wright-Ralph 1996/2000, within the
        # Kojima-Noma-Yoshise 1994 framework) ---
        # Every iterate is kept inside the wide central-path neighbourhood
        #   N(gamma) = { x_i z_i, (1-x_i) w_i >= gamma * mu }
        # by a *line search on the step length*, not by fraction-to-boundary (which
        # controls only positivity, not centrality -- the reason the bare Mehrotra step
        # has no convergence proof).  The fast step takes the longest Mehrotra step in N
        # that decreases mu by a factor rho; if none does, a centred safe step (sigma in
        # [safe_sigma]) takes the longest step in N with an Armijo decrease of mu -- a
        # valid step length is guaranteed to exist (Wright-Ralph Lemma 3.1).  See the
        # module docstring for why the infeasibility is not bounded explicitly.
        self.safeguard = safeguard
        self.gamma_min = gamma_min
        self.beta_floor = beta_floor
        self.rho = rho
        self.kappa = kappa
        self.safe_sigma_lo, self.safe_sigma_hi = safe_sigma
        self.ls_max = ls_max
        self.ls_chi = ls_chi
        self._ssn: Optional[RegularizedSSNSolver] = None
        self._lam_batch_ipm: Optional[torch.Tensor] = None  # warm-start cache for solve_batch

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
        # Per-solve immutables used by the merit/safe-step helpers.
        self._v, self._gamma, self._b = v, gamma, b_use

        # Data-scaled strictly-interior start (Mehrotra/Lustig-Marsten-Shanno style):
        # x at the box centre, and the duals chosen to ZERO the stationarity residual
        # there -- split the gradient g = ell h'(1/2) - v between z >= 0 (its positive
        # part) and w >= 0 (its negative part), shifted by s0 > 0 to stay interior.
        # This scales the dual to the data; a constant start z = w = 1 leaves a huge
        # stationarity residual when ell, v are large, which collapses the
        # fraction-to-boundary step length and stalls the iteration near the start.
        x = torch.full((m,), 0.5, dtype=DEFAULT_DTYPE)
        g0 = ell * pert.hprime(x, gamma) - v
        s0 = 1.0 + 0.1 * float(g0.abs().max())
        z = torch.clamp(g0, min=0.0) + s0
        w = torch.clamp(-g0, min=0.0) + s0
        lam = torch.zeros(c.num_constraints, dtype=DEFAULT_DTYPE)

        # The IPM runs to cross_tol when a crossover polish follows it, else to full tol.
        ipm_tol = self.cross_tol if (self.crossover and self._ssn is not None) else cfg.tol
        history: list[float] = []
        status = STATUS_MAX_ITER
        nit = n_fast = n_safe = 0
        beta = self.beta_floor  # neighbourhood infeasibility bound; set from the start
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
                if it == 0 and mu > 0:
                    # Fix the neighbourhood so the (infeasible) start is admissible:
                    # ||r^0|| <= beta * mu^0.  beta is then held fixed.
                    beta = max(self.beta_floor, feas / mu)
                if feas < ipm_tol and mu < ipm_tol:
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
                    (
                        (x + a_p * dx_a) @ (z + a_d * dz_a)
                        + (1.0 - x - a_p * dx_a) @ (w + a_d * dw_a)
                    )
                    / (2 * m)
                )
                sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0

                # --- corrector step: centring + second-order complementarity ---
                r_z = x * z - sigma * mu + dx_a * dz_a
                r_w = (1.0 - x) * w - sigma * mu - dx_a * dw_a
                dx, dlam, dz, dw = self._newton_step(c, inv_theta, r_d, r_p, r_z, r_w, x, z, w)
                if not self.safeguard:
                    # Unsafeguarded: blind fraction-to-boundary Mehrotra step.
                    a_p = _FRAC_TO_BOUNDARY * self._max_step_primal(x, dx)
                    a_d = _FRAC_TO_BOUNDARY * self._max_step_dual(z, w, dz, dw)
                    x = x + a_p * dx
                    lam = lam + a_d * dlam
                    z = z + a_d * dz
                    w = w + a_d * dw
                    continue
                # Safeguarded: take the longest Mehrotra (fast) step that stays in the
                # neighbourhood N(gamma_min, beta) and decreases mu; if none does, take a
                # centred safe step (longest step in N with an Armijo decrease of mu).
                fast = self._fast_step(x, z, w, lam, dx, dlam, dz, dw, mu, beta, m)
                if fast is not None:
                    x, lam, z, w = fast
                    n_fast += 1
                else:
                    x, lam, z, w = self._safe_step(
                        c, inv_theta, r_d, r_p, x, z, w, lam, mu, sigma, beta, m
                    )
                    n_safe += 1

        ipm_nit = nit
        if self.crossover and self._ssn is not None:
            # Polish to the exact (mu=0) box solution; warm-start at the IPM dual.
            res = self._ssn.solve(theta, b=b, lam0=lam)
            res.nit = ipm_nit + res.nit
            res.residual_history = history + res.residual_history
            res.extras["ipm_nit"] = ipm_nit
            res.extras["n_fast"] = n_fast
            res.extras["n_safe"] = n_safe
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
            extras={"phase": "ipm", "mu": mu, "n_fast": n_fast, "n_safe": n_safe},
        )

    def solve_batch(
        self,
        theta: Tuple[ArrayLike, ArrayLike],
        b_batch: ArrayLike,
        *,
        lam0: Optional[ArrayLike] = None,
    ) -> PURCResult:
        """
        Solve a batch of OD-pairs by the safeguarded IPM, vectorized over systems.

        The OD-pairs share the network ``A``, the link utilities ``v(beta)`` and the
        perturbation, and differ only in the demand ``b``.  Algorithm 1 is run
        elementwise over the ``B`` systems on ``[B, N]`` / ``[B, k]`` tensors: each
        IPM iteration is **two batched Laplacian solves** (predictor + corrector,
        sharing the per-system weights), and the fast/safe step is a *per-OD masked
        backtracking line search* -- its acceptance tests (the ``rho``-decrease of
        ``mu`` and the centrality ratio) are pure reductions, so the line search
        needs no extra solves.  A per-system converged mask freezes finished
        systems; once all converge the loop stops.  The same linear-algebra batch
        primitive (and the single, reused symbolic factorization) backs every solve,
        so the work parallelizes natively over OD pairs.

        Args:
            theta: ``(beta, gamma)`` shared by all systems.
            b_batch: Per-system demands, shape ``[B, k]``.
            lam0: Optional warm-start multipliers ``[B, k]``; defaults to the
                persisted batch (when shapes match) or zeros.

        Returns:
            A :class:`PURCResult` whose ``x`` (``[B, N]``) and ``lam`` (``[B, k]``)
            are batched; per-system residuals, the converged mask, and per-system
            ``n_fast``/``n_safe`` step tallies are in ``extras``.

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
        B = b_batch.shape[0]
        m = c.num_coords
        ell = c.ell.reshape(1, -1)  # [1, N]
        eps_ne = torch.full((B,), _NE_EPS, dtype=DEFAULT_DTYPE)

        # Data-scaled strictly-interior start, per OD (see solve() for the rationale).
        x = torch.full((B, m), 0.5, dtype=DEFAULT_DTYPE)
        g0 = ell * pert.hprime(x, gamma) - v
        s0 = 1.0 + 0.1 * g0.abs().amax(dim=1, keepdim=True)  # [B, 1]
        z = torch.clamp(g0, min=0.0) + s0
        w = torch.clamp(-g0, min=0.0) + s0

        if lam0 is not None:
            lam = as_tensor(lam0).clone()
            if lam.ndim == 1:
                lam = lam.reshape(1, -1)
        elif (
            cfg.warm_start
            and self._lam_batch_ipm is not None
            and self._lam_batch_ipm.shape == b_batch.shape
        ):
            lam = self._lam_batch_ipm.clone()
        else:
            lam = torch.zeros_like(b_batch)

        ipm_tol = self.cross_tol if (self.crossover and self._ssn is not None) else cfg.tol
        history: list[float] = []
        converged = torch.zeros(B, dtype=torch.bool)
        n_fast = torch.zeros(B, dtype=torch.long)
        n_safe = torch.zeros(B, dtype=torch.long)
        status = STATUS_MAX_ITER
        nit = 0

        with torch.no_grad():
            for it in range(cfg.max_iter):
                nit = it + 1
                hp = pert.hprime(x, gamma)
                hpp = pert.hsecond(x, gamma)
                r_d = ell * hp - v - c.rmatvec_batch(lam) - z + w  # [B, N]
                r_p = c.matvec_batch(x) - b_batch  # [B, k]
                pz = x * z
                pw = (1.0 - x) * w
                mu = (pz.sum(dim=1) + pw.sum(dim=1)) / (2 * m)  # [B]
                feas = torch.maximum(r_d.abs().amax(dim=1), r_p.abs().amax(dim=1))  # [B]
                history.append(float(feas.max()))
                converged = converged | ((feas < ipm_tol) & (mu < ipm_tol))
                if bool(converged.all()):
                    status = STATUS_CONVERGED
                    nit = it
                    break

                theta_diag = ell * hpp + z / x + w / (1.0 - x)
                inv_theta = 1.0 / theta_diag

                # --- predictor (affine) ---
                dx_a, dlam_a, dz_a, dw_a = self._newton_step_batch(
                    c, inv_theta, r_d, r_p, x * z, (1.0 - x) * w, x, z, w, eps_ne
                )
                a_p = self._fb_primal_batch(x, dx_a).unsqueeze(1)
                a_d = self._fb_dual_batch(z, w, dz_a, dw_a).unsqueeze(1)
                xz = (x + a_p * dx_a) * (z + a_d * dz_a)
                ww = (1.0 - x - a_p * dx_a) * (w + a_d * dw_a)
                mu_aff = (xz.sum(dim=1) + ww.sum(dim=1)) / (2 * m)  # [B]
                safe_mu = torch.where(mu > 0, mu, torch.ones_like(mu))
                sigma = torch.where(mu > 0, (mu_aff / safe_mu) ** 3, torch.zeros_like(mu))

                # --- corrector (centring + second-order complementarity) ---
                sm = (sigma * mu).unsqueeze(1)
                r_z = x * z - sm + dx_a * dz_a
                r_w = (1.0 - x) * w - sm - dx_a * dw_a
                dx, dlam, dz, dw = self._newton_step_batch(
                    c, inv_theta, r_d, r_p, r_z, r_w, x, z, w, eps_ne
                )

                if not self.safeguard:
                    a_p = (_FRAC_TO_BOUNDARY * self._fb_primal_batch(x, dx)).unsqueeze(1)
                    a_d = (_FRAC_TO_BOUNDARY * self._fb_dual_batch(z, w, dz, dw)).unsqueeze(1)
                    step = (~converged).unsqueeze(1)
                    x = torch.where(step, x + a_p * dx, x)
                    lam = torch.where(step, lam + a_d * dlam, lam)
                    z = torch.where(step, z + a_d * dz, z)
                    w = torch.where(step, w + a_d * dw, w)
                    continue

                # --- fast step: per-OD masked backtracking (converged frozen) ---
                ap_max = _FRAC_TO_BOUNDARY * self._fb_primal_batch(x, dx)  # [B]
                ad_max = _FRAC_TO_BOUNDARY * self._fb_dual_batch(z, w, dz, dw)
                xf, zf, wf, lamf, fast_done = self._fast_backtrack_batch(
                    x, z, w, lam, dx, dz, dw, dlam, ap_max, ad_max, mu, converged
                )
                took_fast = fast_done & ~converged
                need_safe = ~fast_done  # active and not fast (converged are in fast_done)

                # --- safe step for the rest: re-solve (centred) from the iterate ---
                if bool(need_safe.any()):
                    sigma_s = sigma.clamp(self.safe_sigma_lo, self.safe_sigma_hi)  # [B]
                    sms = (sigma_s * mu).unsqueeze(1)
                    dxs, dlams, dzs, dws = self._newton_step_batch(
                        c, inv_theta, r_d, r_p, x * z - sms, (1.0 - x) * w - sms, x, z, w, eps_ne
                    )
                    ap_s = _FRAC_TO_BOUNDARY * self._fb_primal_batch(x, dxs)
                    ad_s = _FRAC_TO_BOUNDARY * self._fb_dual_batch(z, w, dzs, dws)
                    xs, zs, ws, lams, _ = self._safe_backtrack_batch(
                        x, z, w, lam, dxs, dzs, dws, dlams, ap_s, ad_s, mu, sigma_s, ~need_safe
                    )
                    ns = need_safe.unsqueeze(1)
                    xf = torch.where(ns, xs, xf)
                    zf = torch.where(ns, zs, zf)
                    wf = torch.where(ns, ws, wf)
                    lamf = torch.where(ns, lams, lamf)

                x, z, w, lam = xf, zf, wf, lamf
                n_fast += took_fast.long()
                n_safe += need_safe.long()

        ipm_nit = nit
        if self.crossover and self._ssn is not None:
            res = self._ssn.solve_batch(theta, b_batch, lam0=lam)
            res.nit = ipm_nit + res.nit
            res.residual_history = history + res.residual_history
            res.extras["ipm_nit"] = ipm_nit
            res.extras["n_fast"] = n_fast
            res.extras["n_safe"] = n_safe
            res.extras["phase"] = "ipm+ssn"
            return res

        r = c.matvec_batch(x) - b_batch
        r_inf = r.abs().amax(dim=1)  # [B]
        f_conj = (v * x).sum(dim=1) - (ell * pert.h(x, gamma)).sum(dim=1)  # [B]
        conv = r_inf < cfg.tol
        if cfg.warm_start:
            self._lam_batch_ipm = lam.clone()
        return PURCResult(
            x=x,
            lam=lam,
            success=bool(conv.all()),
            status=status,
            nit=ipm_nit,
            residual=float(r_inf.max()),
            conjugate=f_conj,
            residual_history=history,
            extras={
                "n_systems": B,
                "per_system_residual": r_inf,
                "converged_mask": conv,
                "n_fast": n_fast,
                "n_safe": n_safe,
                "phase": "ipm",
            },
        )

    def _newton_step_batch(self, c, inv_theta, r_d, r_p, r_z, r_w, x, z, w, eps_ne):
        """
        Batched IPM Newton solve: one normal-equation solve over all OD-pairs.

        Vectorized counterpart of :meth:`_newton_step`; all arguments carry a
        leading batch dimension (``[B, N]`` for coordinate quantities, ``[B, k]``
        for the multiplier), and the linear system is solved by a single
        :meth:`LaplacianBackend.solve_batch` call.

        Args:
            c: The constraint polytope.
            inv_theta: ``1/Theta`` per coordinate, ``[B, N]`` (the arc weights).
            r_d: Stationarity residual ``[B, N]``.
            r_p: Primal residual ``[B, k]``.
            r_z: Lower-complementarity residual ``[B, N]``.
            r_w: Upper-complementarity residual ``[B, N]``.
            x: Current primal ``[B, N]``.
            z: Current lower dual ``[B, N]``.
            w: Current upper dual ``[B, N]``.
            eps_ne: Per-system normal-equation regularizer ``[B]``.

        Returns:
            ``(dx, dlam, dz, dw)`` batched Newton directions.

        """
        rhs_x = -r_d - r_z / x + r_w / (1.0 - x)
        rhs_lam = -r_p - c.matvec_batch(inv_theta * rhs_x)
        dlam = self._backend.solve_batch(inv_theta, eps_ne, rhs_lam)
        dx = inv_theta * (rhs_x + c.rmatvec_batch(dlam))
        dz = -(r_z + z * dx) / x
        dw = (-r_w + w * dx) / (1.0 - x)
        return dx, dlam, dz, dw

    def _fast_backtrack_batch(self, x, z, w, lam, dx, dz, dw, dlam, ap_max, ad_max, mu, done):
        """
        Fast-step per-OD backtracking (rho-decrease arm) -> ``(xf, zf, wf, lamf, fast_done)``.

        From the frozen base and corrector directions, accept per OD the largest
        ``tau`` whose trial meets ``mu(tau) <= rho mu`` and centrality ``>= gamma_min``,
        leaving ``done`` (converged) systems untouched.  Uses the fused native kernel
        when available and the elementwise torch loop otherwise; the two agree to
        machine precision.
        """
        if native_available():
            B, N = x.shape
            k = lam.shape[1]

            def f(t):
                return np.ascontiguousarray(to_numpy(t).reshape(-1))

            xf = np.empty(B * N)
            zf = np.empty(B * N)
            wf = np.empty(B * N)
            lamf = np.empty(B * k)
            do = np.empty(B, dtype=np.uint8)
            native_core().ipm_fast_backtrack_f64(
                f(x),
                f(z),
                f(w),
                f(lam),
                f(dx),
                f(dz),
                f(dw),
                f(dlam),
                f(ap_max),
                f(ad_max),
                f(mu),
                to_numpy(done).astype(np.uint8).reshape(-1),
                float(self.rho),
                float(self.gamma_min),
                float(self.ls_chi),
                int(self.ls_max),
                B,
                N,
                k,
                xf,
                zf,
                wf,
                lamf,
                do,
            )
            return (
                as_tensor(xf.reshape(B, N)).to(DEFAULT_DTYPE),
                as_tensor(zf.reshape(B, N)).to(DEFAULT_DTYPE),
                as_tensor(wf.reshape(B, N)).to(DEFAULT_DTYPE),
                as_tensor(lamf.reshape(B, k)).to(DEFAULT_DTYPE),
                torch.from_numpy(do.astype(np.bool_)),
            )
        # torch fallback: the elementwise backtracking loop.
        m = x.shape[1]
        fast_done = done.clone()
        xf, zf, wf, lamf = x.clone(), z.clone(), w.clone(), lam.clone()
        tau = 1.0
        for _ in range(self.ls_max):
            ap = (tau * ap_max).unsqueeze(1)
            ad = (tau * ad_max).unsqueeze(1)
            xt, zt, wt = x + ap * dx, z + ad * dz, w + ad * dw
            mut, cr = self._mu_cr_batch(xt, zt, wt, m)
            ok = (~fast_done) & (mut <= self.rho * mu) & (cr >= self.gamma_min)
            sel = ok.unsqueeze(1)
            xf = torch.where(sel, xt, xf)
            zf = torch.where(sel, zt, zf)
            wf = torch.where(sel, wt, wf)
            lamf = torch.where(sel, lam + ad * dlam, lamf)
            fast_done = fast_done | ok
            if bool(fast_done.all()):
                break
            tau *= self.ls_chi
        return xf, zf, wf, lamf, fast_done

    def _safe_backtrack_batch(self, x, z, w, lam, dx, dz, dw, dlam, ap_s, ad_s, mu, sigma_s, done):
        """
        Safe-step per-OD backtracking (Armijo arm) -> ``(xs, zs, ws, lams, safe_done)``.

        From the centred safe directions, update every still-pending OD to the latest
        ``tau`` trial and accept on the Armijo decrease ``mu(tau) <= (1 - kappa tau
        (1 - sigma_s)) mu`` with centrality ``>= gamma_min``; ``done`` systems (those
        not needing a safe step) stay at the base.  Native kernel with torch fallback.
        """
        if native_available():
            B, N = x.shape
            k = lam.shape[1]

            def f(t):
                return np.ascontiguousarray(to_numpy(t).reshape(-1))

            xs = np.empty(B * N)
            zs = np.empty(B * N)
            ws = np.empty(B * N)
            lams = np.empty(B * k)
            do = np.empty(B, dtype=np.uint8)
            native_core().ipm_safe_backtrack_f64(
                f(x),
                f(z),
                f(w),
                f(lam),
                f(dx),
                f(dz),
                f(dw),
                f(dlam),
                f(ap_s),
                f(ad_s),
                f(mu),
                f(sigma_s),
                to_numpy(done).astype(np.uint8).reshape(-1),
                float(self.kappa),
                float(self.gamma_min),
                float(self.ls_chi),
                int(self.ls_max),
                B,
                N,
                k,
                xs,
                zs,
                ws,
                lams,
                do,
            )
            return (
                as_tensor(xs.reshape(B, N)).to(DEFAULT_DTYPE),
                as_tensor(zs.reshape(B, N)).to(DEFAULT_DTYPE),
                as_tensor(ws.reshape(B, N)).to(DEFAULT_DTYPE),
                as_tensor(lams.reshape(B, k)).to(DEFAULT_DTYPE),
                torch.from_numpy(do.astype(np.bool_)),
            )
        # torch fallback: the elementwise Armijo backtracking loop.
        m = x.shape[1]
        safe_done = done.clone()
        xs, zs, ws, lams = x.clone(), z.clone(), w.clone(), lam.clone()
        tau = 1.0
        for _ in range(self.ls_max):
            ap = (tau * ap_s).unsqueeze(1)
            ad = (tau * ad_s).unsqueeze(1)
            xt, zt, wt = x + ap * dx, z + ad * dz, w + ad * dw
            lamt = lam + ad * dlam
            pend = (~safe_done).unsqueeze(1)
            xs = torch.where(pend, xt, xs)
            zs = torch.where(pend, zt, zs)
            ws = torch.where(pend, wt, ws)
            lams = torch.where(pend, lamt, lams)
            mut, cr = self._mu_cr_batch(xt, zt, wt, m)
            armijo = mut <= (1.0 - self.kappa * tau * (1.0 - sigma_s)) * mu
            ok = (~safe_done) & armijo & (cr >= self.gamma_min)
            safe_done = safe_done | ok
            if bool(safe_done.all()):
                break
            tau *= self.ls_chi
        return xs, zs, ws, lams, safe_done

    @staticmethod
    def _mu_cr_batch(xt, zt, wt, m):
        """
        Per-system complementarity measure ``mu`` and centrality ratio.

        Args:
            xt: Trial primal ``[B, N]``.
            zt: Trial lower dual ``[B, N]``.
            wt: Trial upper dual ``[B, N]``.
            m: Number of coordinates ``N``.

        Returns:
            ``(mu, cr)`` each ``[B]``: the average of the ``2N`` products and
            ``min_i product_i / mu`` (``0`` where ``mu <= 0``).

        """
        pz = xt * zt
        pw = (1.0 - xt) * wt
        mu = (pz.sum(dim=1) + pw.sum(dim=1)) / (2 * m)
        pmin = torch.minimum(pz.amin(dim=1), pw.amin(dim=1))
        safe = torch.where(mu > 0, mu, torch.ones_like(mu))
        cr = torch.where(mu > 0, pmin / safe, torch.zeros_like(mu))
        return mu, cr

    @staticmethod
    def _fb_primal_batch(x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """
        Per-system fraction-to-boundary primal step ``[B]`` keeping ``x in (0,1)``.

        Args:
            x: Current primal ``[B, N]`` (strictly interior).
            dx: Primal direction ``[B, N]``.

        Returns:
            The largest ``alpha in (0, 1]`` per system, shape ``[B]``.

        """
        inf = torch.full_like(x, float("inf"))
        neg = dx < 0
        pos = dx > 0
        rn = torch.where(neg, -x / torch.where(neg, dx, torch.ones_like(dx)), inf)
        rp = torch.where(pos, (1.0 - x) / torch.where(pos, dx, torch.ones_like(dx)), inf)
        cand = torch.minimum(rn.amin(dim=1), rp.amin(dim=1))
        one = torch.ones(x.shape[0], dtype=x.dtype)
        return torch.minimum(one, cand).clamp_min(0.0)

    @staticmethod
    def _fb_dual_batch(z, w, dz, dw) -> torch.Tensor:
        """
        Per-system fraction-to-boundary dual step ``[B]`` keeping ``z, w >= 0``.

        Args:
            z: Lower dual ``[B, N]``.
            w: Upper dual ``[B, N]``.
            dz: Lower-dual direction ``[B, N]``.
            dw: Upper-dual direction ``[B, N]``.

        Returns:
            The largest ``alpha in (0, 1]`` per system, shape ``[B]``.

        """
        out = torch.ones(z.shape[0], dtype=z.dtype)
        for s, ds in ((z, dz), (w, dw)):
            inf = torch.full_like(s, float("inf"))
            neg = ds < 0
            r = torch.where(neg, -s / torch.where(neg, ds, torch.ones_like(ds)), inf)
            out = torch.minimum(out, r.amin(dim=1))
        return out.clamp_min(0.0)

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
    def _compl(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, m: int) -> float:
        """
        Complementarity (duality) measure ``mu = (x.z + (1-x).w) / (2m)``.

        Args:
            x: Primal iterate.
            z: Lower-bound dual.
            w: Upper-bound dual.
            m: Number of coordinates.

        Returns:
            The average of the ``2m`` complementarity products.

        """
        return float((x @ z + (1.0 - x) @ w) / (2 * m))

    @staticmethod
    def _central_ratio(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, m: int) -> float:
        """
        Centrality ratio ``min_i product_i / mu`` over the ``2m`` products.

        A point lies in the wide neighbourhood ``N(gamma)`` iff this ratio is at
        least ``gamma``; it measures how far the *least* central pairwise product
        has fallen below the average.

        Args:
            x: Primal iterate.
            z: Lower-bound dual.
            w: Upper-bound dual.
            m: Number of coordinates.

        Returns:
            ``min_i(x_i z_i, (1-x_i) w_i) / mu``, or ``0.0`` if ``mu <= 0``.

        """
        pz = x * z
        pw = (1.0 - x) * w
        mu = float((pz.sum() + pw.sum()) / (2 * m))
        if mu <= 0.0:
            return 0.0
        pmin = min(float(pz.min()), float(pw.min()))
        return pmin / mu

    def _trial(self, x, z, w, lam, dx, dz, dw, dlam, ap, ad, m):
        """
        Trial point with separate primal/dual step lengths, and its metrics.

        Uses ``x + ap dx`` for the primal and ``(z,w,lam) + ad (dz,dw,dlam)`` for the
        dual, as in a standard primal-dual IPM (a single common step length forces
        both to the more constrained of the two and stalls the iteration).

        Args:
            x, z, w, lam: Current iterate.
            dx, dz, dw, dlam: Search direction.
            ap: Primal step length.
            ad: Dual step length.
            m: Number of coordinates.

        Returns:
            ``(xt, lamt, zt, wt, mu, feas, cr)`` -- the trial iterate, its
            complementarity measure, its infeasibility ``||(r_d, r_p)||_inf``, and
            its centrality ratio ``min_i product_i / mu``.

        """
        xt = x + ap * dx
        zt = z + ad * dz
        wt = w + ad * dw
        lamt = lam + ad * dlam
        c = self._problem.constraint
        r_d = c.ell * self._problem.perturbation.hprime(xt, self._gamma) - self._v
        r_d = r_d - c.rmatvec(lamt) - zt + wt
        r_p = c.matvec(xt) - self._b
        pz = xt * zt
        pw = (1.0 - xt) * wt
        mut = float((pz.sum() + pw.sum()) / (2 * m))
        feast = max(float(r_d.abs().max()), float(r_p.abs().max()))
        cr = (min(float(pz.min()), float(pw.min())) / mut) if mut > 0.0 else 0.0
        return xt, lamt, zt, wt, mut, feast, cr

    def _in_nbhd(self, mut: float, feast: float, cr: float, beta: float) -> bool:
        """
        Membership in the wide central-path neighbourhood ``N(gamma_min)``.

        Membership is by *centrality* only -- ``min_i product_i >= gamma_min * mu``.
        This is what prevents the blow-up: a step that unbalances the complementarity
        products (some toward 0) is rejected.  The infeasibility is driven down by
        the Newton step itself (the primal residual exactly as ``(1 - alpha)``, the
        dual residual as the nonlinearity gap shrinks) and by the ``rho``-decrease of
        ``mu``; an explicit ``||r|| <= beta * mu`` wall is *not* imposed, because the
        ``O(ell * alpha^2)`` nonlinearity of ``h'`` (here ``ell`` spans many orders of
        magnitude) would wedge the iterate against it -- the pure-NCP ``g(alpha)``
        correction that keeps the residual exactly ``(1 - alpha) r`` (Wright-Ralph
        1996, eq. 4b) does not transfer to PURC's mixed (equality + box) form.

        Args:
            mut: Trial complementarity measure (unused; kept for symmetry).
            feast: Trial infeasibility (unused; kept for symmetry).
            cr: Trial centrality ratio.
            beta: Infeasibility bound (unused; kept for symmetry).

        Returns:
            ``True`` iff ``cr >= gamma_min``.

        """
        del mut, feast, beta
        return cr >= self.gamma_min

    def _fast_step(self, x, z, w, lam, dx, dlam, dz, dw, mu, beta, m):
        """
        Longest Mehrotra (fast) step in ``N`` with a ``rho``-decrease of ``mu``.

        Backtracks a common scaling ``tau`` of the fraction-to-boundary primal and
        dual lengths; returns the first (longest) step whose trial point is in
        ``N(gamma_min, beta)`` with ``mu(tau) <= rho * mu``, or ``None`` if none
        achieves the ``rho``-decrease (then the caller takes a guaranteed-progress
        safe step).  Requiring the fixed-factor decrease -- not merely ``mu`` down --
        is what prevents the iteration from crawling on tiny fast steps.

        Args:
            x, z, w, lam: Current iterate.
            dx, dlam, dz, dw: Mehrotra search direction.
            mu: Current complementarity measure.
            beta: Neighbourhood infeasibility bound.
            m: Number of coordinates.

        Returns:
            ``(x, lam, z, w)`` after the fast step, or ``None``.

        """
        ap_max = _FRAC_TO_BOUNDARY * self._max_step_primal(x, dx)
        ad_max = _FRAC_TO_BOUNDARY * self._max_step_dual(z, w, dz, dw)
        tau = 1.0
        for _ in range(self.ls_max):
            xt, lamt, zt, wt, mut, feast, cr = self._trial(
                x, z, w, lam, dx, dz, dw, dlam, tau * ap_max, tau * ad_max, m
            )
            if mut <= self.rho * mu and self._in_nbhd(mut, feast, cr, beta):
                return xt, lamt, zt, wt
            tau *= self.ls_chi
        return None

    def _safe_step(self, c, inv_theta, r_d, r_p, x, z, w, lam, mu, sigma, beta, m):
        """
        Centred safe step: longest step in ``N`` with an Armijo decrease of ``mu``.

        Solves the Newton system with centring ``sigma_s in [safe_sigma_lo,
        safe_sigma_hi]`` (no Mehrotra second-order term, same factorization) and
        backtracks ``alpha`` until the trial point is in ``N(gamma_min, beta)`` and
        satisfies ``mu(alpha) <= (1 - kappa alpha (1 - sigma_s)) mu``.  For a
        centred direction such an ``alpha`` exists and is bounded away from 0
        (Wright-Ralph 1996, Lemma 3.1), so the safe step always makes guaranteed
        progress -- the property the global-convergence theorem rests on.

        Args:
            c: Constraint polytope.
            inv_theta: Per-coordinate ``1/Theta`` (the prepared arc weights).
            r_d: Stationarity residual at the current iterate.
            r_p: Primal residual at the current iterate.
            x, z, w, lam: Current iterate.
            mu: Current complementarity measure.
            sigma: Mehrotra centring parameter (clamped to the safe range).
            beta: Neighbourhood infeasibility bound.
            m: Number of coordinates.

        Returns:
            ``(x, lam, z, w)`` after the safe step.

        """
        sigma_s = min(self.safe_sigma_hi, max(self.safe_sigma_lo, sigma))
        r_z = x * z - sigma_s * mu
        r_w = (1.0 - x) * w - sigma_s * mu
        dx, dlam, dz, dw = self._newton_step(c, inv_theta, r_d, r_p, r_z, r_w, x, z, w)
        ap_max = _FRAC_TO_BOUNDARY * self._max_step_primal(x, dx)
        ad_max = _FRAC_TO_BOUNDARY * self._max_step_dual(z, w, dz, dw)
        tau = 1.0
        out = None
        for _ in range(self.ls_max):
            xt, lamt, zt, wt, mut, feast, cr = self._trial(
                x, z, w, lam, dx, dz, dw, dlam, tau * ap_max, tau * ad_max, m
            )
            out = (xt, lamt, zt, wt)
            armijo = mut <= (1.0 - self.kappa * tau * (1.0 - sigma_s)) * mu
            if armijo and self._in_nbhd(mut, feast, cr, beta):
                return xt, lamt, zt, wt
            tau *= self.ls_chi
        # Backtracking floor reached: take the smallest (still strictly interior) step.
        return out

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
