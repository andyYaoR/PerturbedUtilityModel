r"""
Log-barrier primitives for globally convergent PURC continuation methods.

For a fixed ``mu > 0``, the box-barrier forward problem

    min_x F(x; gamma) - v^T x
          - mu sum_i [log(x_i - lo_i) + log(hi_i - x_i)]
    subject to A x = b

has a smooth convex dual.  Given multipliers ``lambda``, every coordinate is
recovered from the strictly monotone scalar equation

    ell_i h'(x_i; gamma) - v_i - (A^T lambda)_i
        - mu/(x_i - lo_i) + mu/(hi_i - x_i) = 0.

The derivative is strictly positive, so safeguarded Newton on the open box gives
a generic recovery routine for all perturbation kernels that implement
``hprime`` and ``hsecond``.  These helpers are intentionally low level: they are
building blocks for a future production continuation solver, not a public solver
class yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..problem import PUMProblem
from ..utils.native import native_available, native_core
from ..utils.torch_compat import DEFAULT_DTYPE, as_tensor, to_numpy
from ..utils.typing import ArrayLike


@dataclass(frozen=True)
class BarrierRecoveryConfig:
    """Numerical controls for fixed-``mu`` barrier coordinate recovery."""

    root_max_iter: int = 80
    root_xtol: float = 1e-13
    endpoint_margin: float = 1e-14

    def __post_init__(self) -> None:
        """Validate root-finder settings."""
        if self.root_max_iter < 1:
            raise ValueError(f"root_max_iter must be >= 1, got {self.root_max_iter}")
        if self.root_xtol <= 0:
            raise ValueError(f"root_xtol must be > 0, got {self.root_xtol}")
        if not (0 < self.endpoint_margin < 0.25):
            raise ValueError(
                "endpoint_margin must be in (0, 0.25), got "
                f"{self.endpoint_margin}"
            )


def recover_barrier_primal(
    problem: PUMProblem,
    v: ArrayLike,
    lam: ArrayLike,
    gamma: Any,
    mu: float,
    config: BarrierRecoveryConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recover interior primal coordinates and Schur weights for fixed ``mu``.

    Args:
        problem: PURC problem with finite box bounds.
        v: Link utilities, shape ``(N,)``.
        lam: Equality multipliers, shape ``(k,)``.
        gamma: Perturbation parameters.
        mu: Positive log-barrier parameter.
        config: Optional root-finder controls.

    Returns:
        ``(x, weight)`` where ``x`` is strictly inside the box and
        ``weight_i = dx_i / d(v_i + A_i^T lambda)``.  The fixed-``mu`` dual
        Hessian is ``A diag(weight) A^T``.

    Raises:
        ValueError: If ``mu <= 0`` or if the box is not finite with positive
            width.

    """
    if mu <= 0:
        raise ValueError(f"mu must be > 0, got {mu}")
    cfg = config or BarrierRecoveryConfig()
    c = problem.constraint
    pert = problem.perturbation
    v_t = as_tensor(v).reshape(-1)
    lam_t = as_tensor(lam).reshape(-1)
    y = v_t + c.rmatvec(lam_t)
    lo = c.lo
    hi = c.hi
    if bool((~torch.isfinite(lo)).any() or (~torch.isfinite(hi)).any()):
        raise ValueError("log-barrier recovery requires finite box bounds")
    gap = hi - lo
    if bool((gap <= 0).any()):
        raise ValueError("log-barrier recovery requires strictly positive box widths")

    tiny = torch.finfo(DEFAULT_DTYPE).eps
    delta = torch.clamp(gap * cfg.endpoint_margin, min=tiny)
    delta = torch.minimum(delta, 0.25 * gap)
    a = lo + delta
    b = hi - delta

    def q_and_qp(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = z - lo
        right = hi - z
        q = c.ell * pert.hprime(z, gamma) - y - mu / left + mu / right
        qp = c.ell * pert.hsecond(z, gamma) + mu / (left * left) + mu / (right * right)
        return q, qp

    eta = y / c.ell
    x0, _ = pert.primal_recovery(eta, lo, hi, gamma)
    midpoint = 0.5 * (a + b)
    on_bound = (x0 <= lo) | (x0 >= hi)
    x = torch.where(on_bound, midpoint, torch.minimum(torch.maximum(x0, a), b))

    # Native per-coordinate root-find: each coordinate converges on its own Newton
    # schedule (no batch-wide width gate), so the stiff-entropy fallback is ~2
    # orders of magnitude faster than the vectorized torch loop below.
    code = getattr(pert, "barrier_kernel_code", -1)
    if native_available() and code >= 0:
        n = int(x.shape[0])
        base = torch.zeros(n, dtype=DEFAULT_DTYPE)
        coeffs = pert.barrier_hp_coeffs(gamma) if code == 4 else np.zeros(0, dtype=np.float64)
        x_out = np.empty(n, dtype=np.float64)
        w_out = np.empty(n, dtype=np.float64)
        native_core().recover_barrier_f64(
            int(code),
            np.ascontiguousarray(coeffs, dtype=np.float64),
            np.ascontiguousarray(to_numpy(base + c.ell)),
            np.ascontiguousarray(to_numpy(y)),
            np.ascontiguousarray(to_numpy(base + as_tensor(lo))),
            np.ascontiguousarray(to_numpy(base + as_tensor(hi))),
            np.ascontiguousarray(to_numpy(x)),
            float(mu),
            float(cfg.endpoint_margin),
            x_out,
            w_out,
            int(cfg.root_max_iter),
            float(cfg.root_xtol),
        )
        return as_tensor(x_out), as_tensor(w_out)

    low = a.clone()
    high = b.clone()

    for _ in range(cfg.root_max_iter):
        q, qp = q_and_qp(x)
        high = torch.where(q > 0, x, high)
        low = torch.where(q <= 0, x, low)

        x_newton = x - q / qp
        out = (~torch.isfinite(x_newton)) | (x_newton <= low) | (x_newton >= high)
        x = torch.where(out, 0.5 * (low + high), x_newton)
        width = high - low
        if bool(torch.all(width <= cfg.root_xtol * (1.0 + gap))):
            break

    x = torch.minimum(torch.maximum(x, a), b)
    _, qp = q_and_qp(x)
    weight = 1.0 / qp
    return x, weight


def barrier_dual_objective(
    problem: PUMProblem,
    v: ArrayLike,
    lam: ArrayLike,
    b: ArrayLike,
    gamma: Any,
    mu: float,
    *,
    x: torch.Tensor | None = None,
    config: BarrierRecoveryConfig | None = None,
) -> float:
    """
    Evaluate the fixed-``mu`` convex dual objective.

    The objective is

        ``-b^T lambda + sum_i max_x [(v_i + A_i^T lambda) x
        - ell_i h(x; gamma) + mu log(x-lo_i) + mu log(hi_i-x)]``.

    Args:
        problem: PURC problem.
        v: Link utilities, shape ``(N,)``.
        lam: Equality multipliers, shape ``(k,)``.
        b: Equality right-hand side, shape ``(k,)``.
        gamma: Perturbation parameters.
        mu: Positive log-barrier parameter.
        x: Optional recovered primal for this ``(lam, mu)``.
        config: Optional recovery controls when ``x`` is not supplied.

    Returns:
        The scalar dual objective value as a Python ``float``.

    """
    c = problem.constraint
    v_t = as_tensor(v).reshape(-1)
    lam_t = as_tensor(lam).reshape(-1)
    b_t = as_tensor(b).reshape(-1)
    if x is None:
        x, _ = recover_barrier_primal(problem, v_t, lam_t, gamma, mu, config)
    y = v_t + c.rmatvec(lam_t)
    log_barrier = torch.log(x - c.lo) + torch.log(c.hi - x)
    value = -(b_t @ lam_t) + torch.sum(
        y * x - c.ell * problem.perturbation.h(x, gamma) + mu * log_barrier
    )
    return float(value)
