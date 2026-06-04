r"""
General polytope ``X = {x : A x = b, lo <= x <= hi}`` for arbitrary sparse ``A``.

No structural assumptions on ``A``: it may be full-rank or rank-deficient.  Rank
deficiency (a per-component nullspace, as in node-arc incidence matrices) needs
no special handling here -- the SSN solver's ``eps_k`` regularization makes every
Newton system SPD and bounds the nullspace step, and the recovered primal ``x*``
is gauge-invariant regardless.  The only feasibility precondition we check is
``b in range(A)`` (otherwise the residual can never reach zero).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..utils.sparse import in_range, row_components, to_csr
from ..utils.typing import ArrayLike
from .base import Polytope


class GeneralPolytope(Polytope):
    """
    Polytope with an arbitrary sparse equality matrix ``A``.

    Args:
        A: The ``(k, N)`` equality-constraint matrix (dense, scipy sparse, or
            CPU torch tensor).
        b: Equality right-hand side, shape ``(k,)``.
        lo: Lower box bounds, shape ``(N,)`` or scalar (default ``0``).
        hi: Upper box bounds, shape ``(N,)`` or scalar (default ``1``).
        ell: Positive separable weights ``ell_i``, shape ``(N,)`` or scalar
            (default ``1``).
        validate: If ``True``, check ``b in range(A)`` and ``ell > 0`` on
            construction and raise on violation.

    Raises:
        ValueError: If shapes are inconsistent, ``ell`` is not positive, or
            (when ``validate``) ``b`` is not in ``range(A)``.

    """

    is_incidence = False

    def __init__(
        self,
        A,
        b: ArrayLike,
        lo: ArrayLike = 0.0,
        hi: ArrayLike = 1.0,
        ell: ArrayLike = 1.0,
        *,
        validate: bool = True,
    ) -> None:
        self._A = to_csr(A)
        k, n = self._A.shape
        self._b = np.broadcast_to(np.asarray(b, dtype=float), (k,)).copy()
        self._lo = np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy()
        self._hi = np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy()
        self._ell = np.broadcast_to(np.asarray(ell, dtype=float), (n,)).copy()

        if np.any(self._lo > self._hi):
            raise ValueError("each lo must be <= the corresponding hi")
        if np.any(self._ell <= 0):
            raise ValueError("separable weights ell must be strictly positive")

        self._n_components: Optional[int] = None
        self._components: Optional[np.ndarray] = None

        if validate and not in_range(self._A, self._b):
            raise ValueError(
                "b is not in range(A): the equality system A x = b is infeasible, "
                "so the SSN residual can never reach zero. Pass validate=False to "
                "skip this check."
            )

    # -- Polytope interface -------------------------------------------------

    @property
    def A(self) -> sp.csr_matrix:
        """The ``(k, N)`` equality-constraint matrix (CSR)."""
        return self._A

    @property
    def b(self) -> np.ndarray:
        """The equality right-hand side, shape ``(k,)``."""
        return self._b

    @property
    def lo(self) -> np.ndarray:
        """Lower box bounds, shape ``(N,)``."""
        return self._lo

    @property
    def hi(self) -> np.ndarray:
        """Upper box bounds, shape ``(N,)``."""
        return self._hi

    @property
    def ell(self) -> np.ndarray:
        """Positive separable weights, shape ``(N,)``."""
        return self._ell

    @property
    def num_coords(self) -> int:
        """Number of decision coordinates ``N``."""
        return self._A.shape[1]

    @property
    def num_constraints(self) -> int:
        """Number of equality rows ``k``."""
        return self._A.shape[0]

    @property
    def n_components(self) -> int:
        """Number of row-connected components (lazily computed and cached)."""
        if self._n_components is None:
            self._n_components, self._components = row_components(self._A)
        return self._n_components

    @property
    def component_labels(self) -> np.ndarray:
        """Per-row component labels, shape ``(k,)`` (lazily computed)."""
        if self._components is None:
            self._n_components, self._components = row_components(self._A)
        return self._components
