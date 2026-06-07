r"""
Abstract base class for the constraint polytope.

The feasible set is ``X = {x in R^N : A x = b, lo <= x <= hi}`` with ``A`` an
``(k, N)`` sparse matrix.  The perturbation weights ``ell_i > 0`` (link lengths
in the route-choice model) live here too, since they scale the separable kernel
``F(x) = sum_i ell_i h(x_i)`` and hence the Newton edge weights.

Subclasses specialize the linear algebra:
  * :class:`GeneralPolytope` -- arbitrary sparse ``A`` (Newton matrix routed to
    LaplacianSolve's SDDM/SPD solvers), and
  * :class:`IncidencePolytope` -- node-arc incidence ``A`` (Newton matrix is a
    weighted graph Laplacian, routed to ``PURCLaplacianSolver``).

Rank-deficient ``A`` (per-component nullspace) makes the dual multipliers
``lambda`` determined only up to a per-component constant; the gauge is fixed in
``constraints/normalization.py`` (M2).  The testable invariant is that the
*primal* ``x*`` is gauge-invariant even though ``lambda`` is not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..utils.typing import ArrayLike, SparseMatrix


class Polytope(ABC):
    """
    Interface for ``X = {A x = b, lo <= x <= hi}`` with separable weights ``ell``.

    Attributes:
        is_incidence: ``True`` if ``A`` is a node-arc incidence matrix (enables
            the network Laplacian fast path).

    """

    is_incidence: bool = False

    @property
    @abstractmethod
    def A(self) -> SparseMatrix:
        """The ``(k, N)`` equality-constraint matrix (sparse)."""

    @property
    @abstractmethod
    def b(self) -> ArrayLike:
        """The equality right-hand side, shape ``(k,)``."""

    @property
    @abstractmethod
    def lo(self) -> ArrayLike:
        """Lower box bounds, shape ``(N,)``."""

    @property
    @abstractmethod
    def hi(self) -> ArrayLike:
        """Upper box bounds, shape ``(N,)``."""

    @property
    @abstractmethod
    def ell(self) -> ArrayLike:
        """Positive separable weights ``ell_i``, shape ``(N,)``."""

    @property
    @abstractmethod
    def num_coords(self) -> int:
        """Number of decision coordinates ``N``."""

    @property
    @abstractmethod
    def num_constraints(self) -> int:
        """Number of equality rows ``k``."""

    @property
    @abstractmethod
    def n_components(self) -> int:
        """
        Number of connected components of ``A`` (its equality-row nullspace
        dimension), i.e. how many reference multipliers must be pinned.
        """

    def matvec(self, x: ArrayLike) -> ArrayLike:
        """
        Apply ``A`` to a coordinate vector.

        Args:
            x: Vector of shape ``(N,)``.

        Returns:
            ``A @ x`` of shape ``(k,)``.

        """
        return self.A @ x

    def rmatvec(self, lam: ArrayLike) -> ArrayLike:
        """
        Apply ``A^T`` to a multiplier vector.

        Args:
            lam: Vector of shape ``(k,)``.

        Returns:
            ``A^T @ lam`` of shape ``(N,)``.

        """
        return self.A.T @ lam
