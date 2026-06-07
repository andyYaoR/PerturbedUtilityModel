"""
Euclidean projections of the sieve coefficients ``gamma`` onto the convexity set.

Two specifications of ``Gamma`` are supported (the paper offers both):

* **Nonnegative** ``Gamma = R_+^{L-2}`` -- conservative; projection clips to ``0``.
* **Bernstein polyhedron** ``Gamma_B = {gamma : M gamma >= -1}`` -- the inside
  Bernstein approximation of ``{h''(.;gamma) >= 0 on [0,1]}``; admits some negative
  ``gamma_l``.  The projection is a tiny QP (dimension ``L-2``), solved in numpy
  via SLSQP and bridged back to torch.

Inputs and outputs are torch tensors (the package is torch-native); ``beta`` is
unconstrained, so only ``gamma`` is projected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from ...static_purc.perturbations._bernstein import bernstein_matrix, is_convex
from ...static_purc.utils.torch_compat import DEFAULT_DTYPE, as_tensor, to_numpy


def project_nonneg(gamma) -> torch.Tensor:
    """
    Project ``gamma`` onto ``R_+^{L-2}`` (clip negatives to zero).

    Args:
        gamma: Sieve coefficients, shape ``[L-2]``.

    Returns:
        ``max(gamma, 0)`` as a torch tensor.

    """
    return torch.clamp(as_tensor(gamma).to(DEFAULT_DTYPE), min=0.0)


def project_bernstein(gamma, basis=None) -> torch.Tensor:
    """
    Euclidean projection onto the Bernstein convexity set.

    In the monomial basis this is ``Gamma_B = {g : M g >= -1}``.  With a non-monomial
    ``basis`` the coefficient is ``c`` and the same convexity polyhedron pulls back to
    ``{c : (M T) c >= -1}`` (the constraint matrix is ``basis.constraint_matrix(M)``);
    feasibility is always tested on the monomial image ``gamma = T c``.  Solves
    ``min 0.5 ||x - x0||^2`` subject to the (possibly pulled-back) constraint via
    SLSQP; already-feasible inputs are returned unchanged.  On a QP failure it falls
    back to the always-feasible nonnegative projection (monomial) or to ``0`` (which
    maps to ``gamma = 0``, the convex quadratic baseline, feasible in any basis).

    Args:
        gamma: Sieve coefficients in the active basis, shape ``[L-2]``.
        basis: Optional :class:`SieveBasis`; ``None`` / monomial keeps the original
            monomial-space projection.

    Returns:
        The projection as a torch tensor (feasible: ``is_convex(T x)`` holds).

    """
    mono = basis is None or basis.is_monomial
    x0 = to_numpy(as_tensor(gamma).to(DEFAULT_DTYPE)).ravel()

    def _mono(x):
        return x if mono else to_numpy(basis.to_monomial(x))

    if x0.size == 0 or is_convex(_mono(x0)):
        return as_tensor(x0).to(DEFAULT_DTYPE)
    M = bernstein_matrix(x0.size)
    A = M if mono else basis.constraint_matrix(M)
    cons = {"type": "ineq", "fun": lambda x: A @ x + 1.0, "jac": lambda x: A}
    res = minimize(
        lambda x: 0.5 * float(np.sum((x - x0) ** 2)),
        x0,
        jac=lambda x: x - x0,
        constraints=[cons],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 200},
    )
    out = np.asarray(res.x, dtype=float)
    if not is_convex(_mono(out)):
        out = project_nonneg(torch.as_tensor(x0)).numpy() if mono else np.zeros_like(x0)
    return as_tensor(out).to(DEFAULT_DTYPE)


@dataclass
class GammaProjection:
    """
    A selectable projection onto the sieve convexity set.

    Args:
        kind: ``"nonneg"`` for ``R_+^{L-2}`` or ``"bernstein"`` for ``Gamma_B``.
        basis: Optional :class:`SieveBasis`.  When non-monomial, the coefficient is
            ``c`` and the projection runs in ``c``-space (Bernstein pulled back).
            ``"nonneg"`` is only defined in the monomial basis (clipping ``c >= 0`` is
            not ``gamma >= 0`` otherwise), so combining it with a non-monomial basis
            is rejected.

    """

    kind: str = "bernstein"
    basis: object = None

    def __post_init__(self) -> None:
        """Reject ``nonneg`` in a non-monomial basis (clipping ``c`` is not ``gamma>=0``)."""
        if self.kind == "nonneg" and self.basis is not None and not self.basis.is_monomial:
            raise ValueError(
                "nonneg projection is only defined in the monomial basis; "
                "use kind='bernstein' with an orthonormal basis"
            )

    def __call__(self, gamma) -> torch.Tensor:
        """
        Project ``gamma`` (or basis ``c``) onto the configured set.

        Args:
            gamma: Sieve coefficients in the active basis, shape ``[L-2]``.

        Returns:
            The projected coefficients (torch).

        Raises:
            ValueError: If ``kind`` is unknown.

        """
        if self.kind == "nonneg":
            return project_nonneg(gamma)
        if self.kind == "bernstein":
            return project_bernstein(gamma, basis=self.basis)
        raise ValueError(f"unknown projection kind {self.kind!r}")

    def is_feasible(self, gamma) -> bool:
        """
        Whether ``gamma`` (or basis ``c``) lies in the configured set.

        Args:
            gamma: Sieve coefficients in the active basis, shape ``[L-2]``.

        Returns:
            ``True`` if feasible.

        """
        g = to_numpy(as_tensor(gamma).to(DEFAULT_DTYPE)).ravel()
        if self.kind == "nonneg":
            return bool(np.all(g >= 0.0))
        mono = self.basis is None or self.basis.is_monomial
        return is_convex(g if mono else to_numpy(self.basis.to_monomial(g)))
