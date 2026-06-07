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


def project_bernstein(gamma) -> torch.Tensor:
    """
    Euclidean projection of ``gamma`` onto ``Gamma_B = {g : M g >= -1}``.

    Solves ``min_g 0.5 ||g - gamma||^2  s.t.  M g >= -1`` via SLSQP.  Already-feasible
    inputs are returned unchanged; on a (tolerance-level) QP failure it falls back
    to the always-feasible nonnegative projection.

    Args:
        gamma: Sieve coefficients, shape ``[L-2]``.

    Returns:
        The projection as a torch tensor (feasible: ``is_convex`` holds).

    """
    g0 = to_numpy(as_tensor(gamma).to(DEFAULT_DTYPE)).ravel()
    if g0.size == 0 or is_convex(g0):
        return as_tensor(g0).to(DEFAULT_DTYPE)
    M = bernstein_matrix(g0.size)
    cons = {"type": "ineq", "fun": lambda g: M @ g + 1.0, "jac": lambda g: M}
    res = minimize(
        lambda g: 0.5 * float(np.sum((g - g0) ** 2)),
        g0,
        jac=lambda g: g - g0,
        constraints=[cons],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 200},
    )
    out = np.asarray(res.x, dtype=float)
    if not is_convex(out):
        out = project_nonneg(torch.as_tensor(g0)).numpy()
    return as_tensor(out).to(DEFAULT_DTYPE)


@dataclass
class GammaProjection:
    """
    A selectable projection onto the sieve convexity set.

    Args:
        kind: ``"nonneg"`` for ``R_+^{L-2}`` or ``"bernstein"`` for ``Gamma_B``.

    """

    kind: str = "bernstein"

    def __call__(self, gamma) -> torch.Tensor:
        """
        Project ``gamma`` onto the configured set.

        Args:
            gamma: Sieve coefficients, shape ``[L-2]``.

        Returns:
            The projected coefficients (torch).

        Raises:
            ValueError: If ``kind`` is unknown.

        """
        if self.kind == "nonneg":
            return project_nonneg(gamma)
        if self.kind == "bernstein":
            return project_bernstein(gamma)
        raise ValueError(f"unknown projection kind {self.kind!r}")

    def is_feasible(self, gamma) -> bool:
        """
        Whether ``gamma`` lies in the configured set.

        Args:
            gamma: Sieve coefficients, shape ``[L-2]``.

        Returns:
            ``True`` if feasible.

        """
        g = to_numpy(as_tensor(gamma).to(DEFAULT_DTYPE)).ravel()
        if self.kind == "nonneg":
            return bool(np.all(g >= 0.0))
        return is_convex(g)
