"""
The :class:`PUMProblem` couples a perturbation, a constraint polytope, and the
linear-utility specification into a single forward problem.

This is the *only* object that binds the three otherwise-independent registries
(perturbations / constraints / solvers) together.  It holds the structural data
that is fixed across an estimation sweep; the parameters ``theta = (beta, gamma)``
vary and are supplied per :meth:`solve` call so the persistent solver handle and
warm-started multipliers can be reused (see ``solvers/base.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .utils.typing import ArrayLike, SparseMatrix

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .constraints.base import Polytope
    from .perturbations.base import SeparablePerturbation


class PUMProblem:
    """
    A perturbed-utility forward problem ``min_x F(x;gamma) - v(beta)^T x`` over a
    polytope ``X = {Ax = b, l <= x <= u}``.

    The link utility is linear in the coefficients, ``v(beta) = Z @ beta``.  When
    ``Z`` is omitted the utility *is* ``beta`` directly (``Z = I``), which is the
    common case for a small hand-built problem or a fixed utility vector.

    Args:
        perturbation: A :class:`~purcsolver.perturbations.base.SeparablePerturbation`.
        constraint: A :class:`~purcsolver.constraints.base.Polytope`.
        Z: Optional ``(N, K)`` link-attribute matrix mapping ``beta`` to link
            utilities ``v = Z @ beta``.  Dense array or scipy sparse.  ``None``
            means ``v = beta`` (so ``K = N``).

    Raises:
        ValueError: If ``Z``'s row count does not match the constraint dimension.

    """

    def __init__(
        self,
        perturbation: "SeparablePerturbation",
        constraint: "Polytope",
        Z: Optional[SparseMatrix] = None,
    ) -> None:
        self.perturbation = perturbation
        self.constraint = constraint
        self.Z = Z
        if Z is not None and Z.shape[0] != constraint.num_coords:
            raise ValueError(
                f"Z has {Z.shape[0]} rows but the constraint has "
                f"{constraint.num_coords} coordinates."
            )

    @property
    def num_coords(self) -> int:
        """Number of decision coordinates ``N`` (links / variables)."""
        return self.constraint.num_coords

    @property
    def num_params(self) -> int:
        """Number of utility coefficients ``K`` (``= N`` when ``Z`` is ``None``)."""
        return self.constraint.num_coords if self.Z is None else self.Z.shape[1]

    def utility(self, beta: ArrayLike) -> ArrayLike:
        """
        Compute the link utility vector ``v(beta) = Z @ beta``.

        Args:
            beta: Utility coefficients, shape ``(K,)``.

        Returns:
            Link utilities ``v``, shape ``(N,)``.

        Raises:
            ValueError: If ``beta`` has the wrong length.

        """
        beta = np.asarray(beta, dtype=float)
        if self.Z is None:
            if beta.shape[0] != self.num_coords:
                raise ValueError(
                    f"beta has length {beta.shape[0]} but Z is None so it must "
                    f"equal the coordinate count {self.num_coords}."
                )
            return beta
        if beta.shape[0] != self.num_params:
            raise ValueError(f"beta has length {beta.shape[0]} but Z expects {self.num_params}.")
        return np.asarray(self.Z @ beta, dtype=float).ravel()
