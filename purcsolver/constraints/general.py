r"""
General polytope ``X = {x : A x = b, lo <= x <= hi}`` for arbitrary sparse ``A``.

No structural assumptions on ``A``: it may be full-rank or rank-deficient.  Rank
deficiency (a per-component nullspace, as in node-arc incidence matrices) needs
no special handling here -- the SSN solver's ``eps_k`` regularization makes every
Newton system SPD and bounds the nullspace step, and the recovered primal ``x*``
is gauge-invariant regardless.  The only feasibility precondition we check is
``b in range(A)`` (otherwise the residual can never reach zero).

The matrix is held as a SciPy CSR (for the SciPy-only consumers: the CSC pattern
build, feasibility, the cvxpy oracle) plus a :class:`~purcsolver.utils.spmv.CSRMatVec`
for ``A`` and ``A^T`` (the native CSR SpMV used for the hot-path matvecs).  Node-arc
incidence structure is auto-detected so the solver can route to LaplacianSolve's
fast ``PURCLaplacianSolver`` network path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from ..utils.sparse import detect_incidence as _detect_incidence
from ..utils.sparse import in_range, row_components, to_csr
from ..utils.spmv import CSRMatVec
from ..utils.torch_compat import DEFAULT_DTYPE, as_tensor, to_numpy
from ..utils.typing import ArrayLike
from .base import Polytope


class GeneralPolytope(Polytope):
    """
    Polytope with an arbitrary sparse equality matrix ``A``.

    Args:
        A: The ``(k, N)`` equality-constraint matrix (dense, scipy sparse, numpy,
            or CPU torch tensor / sparse tensor).
        b: Equality right-hand side, shape ``(k,)``.
        lo: Lower box bounds, shape ``(N,)`` or scalar (default ``0``).
        hi: Upper box bounds, shape ``(N,)`` or scalar (default ``1``).
        ell: Positive separable weights ``ell_i``, shape ``(N,)`` or scalar
            (default ``1``).
        validate: If ``True``, check ``b in range(A)`` and ``ell > 0`` on
            construction and raise on violation.
        detect_incidence: If ``True`` (default), auto-detect node-arc incidence
            structure so the solver can route to LaplacianSolve's fast
            ``PURCLaplacianSolver`` network path.

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
        detect_incidence: bool = True,
    ) -> None:
        self._A_scipy = to_csr(_coerce_to_scipy(A))
        k, n = self._A_scipy.shape
        self._dtype = DEFAULT_DTYPE
        # Auto-detect node-arc incidence -> enables the PURCLaplacianSolver path.
        self._edges: Optional[np.ndarray] = None
        self.is_incidence = False
        if detect_incidence:
            self.is_incidence, self._edges = _detect_incidence(self._A_scipy)
        # Native CSR SpMV for the hot-path matvecs (A x_hat and A^T lambda).
        self._mv = CSRMatVec(self._A_scipy)
        self._mvT = CSRMatVec(self._A_scipy.T.tocsr())

        self._b = as_tensor(b).reshape(-1).broadcast_to((k,)).clone()
        self._lo = _broadcast_box(lo, n)
        self._hi = _broadcast_box(hi, n)
        self._ell = _broadcast_box(ell, n)

        if bool((self._lo > self._hi).any()):
            raise ValueError("each lo must be <= the corresponding hi")
        if bool((self._ell <= 0).any()):
            raise ValueError("separable weights ell must be strictly positive")

        self._n_components: Optional[int] = None
        self._components: Optional[np.ndarray] = None

        if validate and not in_range(self._A_scipy, self._b.detach().cpu().numpy()):
            raise ValueError(
                "b is not in range(A): the equality system A x = b is infeasible, "
                "so the SSN residual can never reach zero. Pass validate=False to "
                "skip this check."
            )

    # -- Polytope interface -------------------------------------------------

    @property
    def A(self) -> sp.csr_matrix:
        """The ``(k, N)`` equality-constraint matrix as a SciPy CSR matrix."""
        return self._A_scipy

    @property
    def b(self) -> torch.Tensor:
        """The equality right-hand side, shape ``(k,)``."""
        return self._b

    @property
    def lo(self) -> torch.Tensor:
        """Lower box bounds, shape ``(N,)``."""
        return self._lo

    @property
    def hi(self) -> torch.Tensor:
        """Upper box bounds, shape ``(N,)``."""
        return self._hi

    @property
    def ell(self) -> torch.Tensor:
        """Positive separable weights, shape ``(N,)``."""
        return self._ell

    @property
    def num_coords(self) -> int:
        """Number of decision coordinates ``N``."""
        return self._A_scipy.shape[1]

    @property
    def num_constraints(self) -> int:
        """Number of equality rows ``k``."""
        return self._A_scipy.shape[0]

    @property
    def edges(self) -> Optional[np.ndarray]:
        """The ``(N, 2)`` edge endpoints if ``A`` is incidence, else ``None``."""
        return self._edges

    @property
    def n_nodes(self) -> int:
        """Number of nodes (equality rows) for the incidence network path."""
        return self._A_scipy.shape[0]

    @property
    def n_components(self) -> int:
        """Number of row-connected components (lazily computed and cached)."""
        if self._n_components is None:
            self._n_components, self._components = row_components(self._A_scipy)
        return self._n_components

    @property
    def component_labels(self) -> np.ndarray:
        """Per-row component labels, shape ``(k,)`` (lazily computed)."""
        if self._components is None:
            self._n_components, self._components = row_components(self._A_scipy)
        return self._components

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ``A`` to a coordinate tensor via the native CSR SpMV.

        The tensor bridges to NumPy zero-copy on CPU; the result is wrapped back
        on the input's device.

        Args:
            x: Tensor of shape ``(N,)``.

        Returns:
            ``A @ x`` of shape ``(k,)``.

        """
        xt = as_tensor(x, dtype=self._dtype)
        return as_tensor(self._mv.matvec(to_numpy(xt)), device=xt.device)

    def rmatvec(self, lam: torch.Tensor) -> torch.Tensor:
        """
        Apply ``A^T`` to a multiplier tensor via the native CSR SpMV.

        Args:
            lam: Tensor of shape ``(k,)``.

        Returns:
            ``A^T @ lam`` of shape ``(N,)``.

        """
        lt = as_tensor(lam, dtype=self._dtype)
        return as_tensor(self._mvT.matvec(to_numpy(lt)), device=lt.device)


def _coerce_to_scipy(A):
    """
    Coerce a torch (sparse or dense) / numpy / scipy matrix to something
    :func:`to_csr` accepts.

    Args:
        A: The input matrix in any supported form.

    Returns:
        A scipy sparse matrix or dense numpy array.

    """
    if isinstance(A, torch.Tensor):
        if A.layout != torch.strided:
            A = A.to_dense()
        return A.detach().cpu().numpy()
    return A


def _broadcast_box(value: ArrayLike, n: int) -> torch.Tensor:
    """
    Broadcast a scalar / vector box parameter to a length-``n`` tensor.

    Args:
        value: Scalar or length-``n`` array-like.
        n: Target length.

    Returns:
        A length-``n`` float64 tensor.

    """
    t = as_tensor(value).reshape(-1)
    if t.numel() == 1:
        return t.expand(n).clone()
    return t.broadcast_to((n,)).clone()
