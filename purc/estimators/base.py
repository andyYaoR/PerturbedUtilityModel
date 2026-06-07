"""
Estimator interface and the generic data containers it consumes and returns.

These types are deliberately model- and estimator-agnostic: a
:class:`SimulatedData` bundle (per-OD link counts and empirical frequencies) is
produced by a data-generating process (e.g.
:func:`purc.static_purc.dgp.simulate_dataset`) or read from real data, and an
:class:`Estimator` maps it to an :class:`EstimationResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class SimulatedData:
    """
    Observed (or simulated) route-choice data for a set of OD pairs.

    All fields are torch tensors (float64, CPU by default -- the package is
    torch-native) sharing the network's link ordering.  Counts and frequencies are
    per OD pair ``b`` and link ``(i, j)``.

    Attributes:
        n_counts: Link traversal counts ``n_{(i,j),b}``, shape ``[B, N]``.
        ybar: Empirical link frequencies ``n / D``, shape ``[B, N]``.
        D: Number of trips ``D_b`` per OD pair, shape ``[B]``.
        b_batch: Per-OD demand vectors, shape ``[B, k]`` (``k`` = #nodes).
        xstar0: The true predicted flows ``x*_b(theta_0)`` used to simulate, shape
            ``[B, N]`` (``None`` for real data).
        meta: Free-form provenance (e.g. the true ``theta_0``, seed, network name).

    """

    n_counts: torch.Tensor
    ybar: torch.Tensor
    D: torch.Tensor
    b_batch: torch.Tensor
    xstar0: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_systems(self) -> int:
        """Number of OD pairs ``B``."""
        return int(self.n_counts.shape[0])


@dataclass
class EstimationResult:
    """
    Outcome of fitting an estimator.

    Attributes:
        theta_hat: The fitted parameter vector (flat ``[beta, gamma]``).
        beta_hat: The fitted utility coefficients.
        gamma_hat: The fitted sieve coefficients.
        objective: Final objective value ``Q_B(theta_hat)``.
        grad: Final (projected) gradient.
        n_outer: Number of outer (projected-gradient) iterations.
        converged: Whether the outer loop met its stopping criterion.
        se: Optional asymptotic standard errors (sandwich), flat like ``theta_hat``.
        extras: Diagnostics (inner-iteration history, condition numbers, etc.).

    """

    theta_hat: torch.Tensor
    beta_hat: torch.Tensor
    gamma_hat: torch.Tensor
    objective: float
    grad: torch.Tensor
    n_outer: int
    converged: bool
    se: Optional[torch.Tensor] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class Estimator(ABC):
    """
    Abstract base for PURC estimators.

    Concrete estimators (e.g. the debiased Fenchel--Young estimator) implement
    :meth:`fit`, mapping a :class:`SimulatedData` bundle to an
    :class:`EstimationResult`.
    """

    @abstractmethod
    def fit(self, data: SimulatedData) -> EstimationResult:
        """
        Fit the estimator to ``data``.

        Args:
            data: The observed/simulated per-OD link data.

        Returns:
            The estimation result.

        """
        raise NotImplementedError
