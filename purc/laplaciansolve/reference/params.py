"""
Solver parameters for the reference approxChol factorization.

Mirrors Julia ``ApproxCholParams(order, stag_test, split, merge)`` from
``Laplacians.jl`` (``src/approxChol.jl``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Elimination orderings supported by the reference port.  ``"deg"`` (eliminate
# the lowest-degree vertex next, adaptive) is the flagship default; ``"given"``
# and ``"wdeg"`` are accepted names but not yet implemented in Stage 0.
_VALID_ORDERS = ("deg", "given", "wdeg")


@dataclass(frozen=True)
class ApproxCholParams:
    """
    Parameters controlling the approximate-Cholesky elimination.

    Attributes:
        order: Elimination ordering, one of ``"deg"`` (adaptive lowest-degree;
            slowest build, fastest solve), ``"given"`` (input order), or
            ``"wdeg"`` (perturbed weighted-degree order).
        stag_test: PCG stagnation window ``k``: PCG stops when
            ``rho[it] > (1 - 1/k) * rho[it - k]``.  ``0`` disables the test.
        split: Number of multi-edge copies each original edge is split into for
            robustness.  ``0`` (default) disables splitting.
        merge: Upper bound on the number of retained multi-edges; recommended to
            equal ``split``.  ``0`` disables merging.

    """

    order: str = "deg"
    stag_test: int = 5
    split: int = 0
    merge: int = 0

    def __post_init__(self) -> None:
        """
        Validate the parameter values.

        Raises:
            ValueError: If ``order`` is not a recognized ordering or if
                ``stag_test`` / ``split`` / ``merge`` are negative.

        """
        if self.order not in _VALID_ORDERS:
            raise ValueError(f"order must be one of {_VALID_ORDERS}, got {self.order!r}")
        if self.stag_test < 0:
            raise ValueError(f"stag_test must be >= 0, got {self.stag_test}")
        if self.split < 0:
            raise ValueError(f"split must be >= 0, got {self.split}")
        if self.merge < 0:
            raise ValueError(f"merge must be >= 0, got {self.merge}")
