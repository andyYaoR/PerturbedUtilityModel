r"""
General polytope ``X = {x : A x = b, lo <= x <= hi}`` for arbitrary sparse ``A``.

No structural assumptions on ``A``: it may be full-rank or rank-deficient.  Rank
deficiency (a per-component nullspace, as in node-arc incidence matrices) needs
no special handling here -- the SSN solver's ``eps_k`` regularization makes every
Newton system SPD and bounds the nullspace step, and the recovered primal ``x*``
is gauge-invariant regardless.  The only feasibility precondition we check is
``b in range(A)`` (otherwise the residual can never reach zero).

The matrix is held in two zero-copy-bridged forms: a SciPy CSR for the SciPy-only
consumers (the CSC pattern build, feasibility, the cvxpy oracle), and torch
sparse-CSR tensors for ``A``/``A^T`` so the torch-native solver's matvecs stay on
the solver's device/dtype.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from ..utils.sparse import in_range, row_components, to_csr
from ..utils.torch_compat import DEFAULT_DTYPE, as_tensor
from ..utils.typing import ArrayLike
from .base import Polytope


def _scipy_to_torch_csr(matrix: sp.csr_matrix, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a SciPy CSR matrix to a torch sparse-CSR tensor.

    Args:
        matrix: A SciPy CSR matrix.
        dtype: Target value dtype.

    Returns:
        The equivalent ``torch.sparse_csr_tensor``.

    """
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64)),
        torch.from_numpy(matrix.indices.astype(np.int64)),
        torch.from_numpy(matrix.data.astype(np.float64)).to(dtype),
        size=matrix.shape,
    )


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
        self._A_scipy = to_csr(_coerce_to_scipy(A))
        k, n = self._A_scipy.shape
        self._dtype = DEFAULT_DTYPE
        self._A = _scipy_to_torch_csr(self._A_scipy, self._dtype)
        self._At = _scipy_to_torch_csr(self._A_scipy.T.tocsr(), self._dtype)

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
        Apply ``A`` to a coordinate tensor (torch sparse matvec).

        Args:
            x: Tensor of shape ``(N,)``.

        Returns:
            ``A @ x`` of shape ``(k,)``.

        """
        return self._A @ as_tensor(x, dtype=self._dtype)

    def rmatvec(self, lam: torch.Tensor) -> torch.Tensor:
        """
        Apply ``A^T`` to a multiplier tensor (torch sparse matvec).

        Args:
            lam: Tensor of shape ``(k,)``.

        Returns:
            ``A^T @ lam`` of shape ``(N,)``.

        """
        return self._At @ as_tensor(lam, dtype=self._dtype)


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
