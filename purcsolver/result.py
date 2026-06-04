"""
Result container for a perturbed-utility forward solve.

:class:`PURCResult` is a SciPy-``OptimizeResult``-style record carrying both the
primal solution and everything downstream consumers need: the dual multipliers
and conjugate value for warm-starting and for assembling Fenchel-Young
estimation gradients later (the FY envelope property means the estimation
gradient is built from ``x*`` and ``F*`` directly -- no ``dx*/dtheta``
sensitivity is required).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils.typing import ArrayLike

# Status codes (mirror SciPy's small-integer convention).
STATUS_CONVERGED = 0
STATUS_MAX_ITER = 1
STATUS_LINESEARCH_FAILED = 2

_STATUS_MESSAGES = {
    STATUS_CONVERGED: "Converged: residual below tolerance.",
    STATUS_MAX_ITER: "Maximum number of Newton iterations reached.",
    STATUS_LINESEARCH_FAILED: "Line search failed to find a sufficient-decrease step.",
}


@dataclass
class PURCResult:
    """
    Outcome of a single forward solve ``min_x F(x;gamma) - v(beta)^T x`` over a
    polytope, via the regularized semismooth Newton method.

    Attributes:
        x: Primal solution ``x*`` (recovered flows / choice probabilities),
            shape ``(N,)`` or ``(B, N)`` for a batch of right-hand sides.
        lam: Dual multipliers ``lambda*`` (node potentials), shape ``(k,)`` or
            ``(B, k)``.  Gauge-fixed deterministically so warm starts and the
            dual objective are reproducible (``x*`` itself is gauge-invariant).
        success: ``True`` iff the solver reached :data:`STATUS_CONVERGED`.
        status: One of the ``STATUS_*`` codes.
        message: Human-readable status description.
        nit: Number of Newton iterations performed (max over a batch).
        residual: Final ``||A x* - b||_inf``.
        conjugate: Convex-conjugate / dual objective value ``F*(v;gamma)`` at the
            solution (the Fenchel-Young term the estimation layer consumes).
        residual_history: ``||r||_inf`` per Newton iteration.
        extras: Diagnostics (active-set sizes, line-search steps, backend
            ``method``/``phase``, per-system convergence for batches, ...).

    """

    x: ArrayLike
    lam: ArrayLike
    success: bool
    status: int
    message: str = ""
    nit: int = 0
    residual: float = float("nan")
    conjugate: Optional[ArrayLike] = None
    residual_history: List[float] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill a default message from the status code when none was provided."""
        if not self.message:
            self.message = _STATUS_MESSAGES.get(self.status, f"status={self.status}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a plain-dict view of the result (for logging / serialization).

        Returns:
            A dictionary with the public result fields.

        """
        return {
            "x": self.x,
            "lam": self.lam,
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "nit": self.nit,
            "residual": self.residual,
            "conjugate": self.conjugate,
            "residual_history": list(self.residual_history),
            "extras": dict(self.extras),
        }
