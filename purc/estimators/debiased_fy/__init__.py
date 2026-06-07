"""
The debiased Fenchel--Young estimator for the static PURC model.

Implements the estimator of the paper: a debiased FY loss whose per-OD primal
term replaces the biased plug-in ``y^l`` with falling-factorial U-statistics
(unbiased for ``(x*)^l``), minimized over ``R^K x Gamma`` by projected gradient
descent (the inner forward solves use the batched IPM).  Exposes the loss, the
optimizer (the :class:`~purc.estimators.base.Estimator`), the Gamma projections,
and the sandwich-variance utilities.
"""

from __future__ import annotations

from .basis import SieveBasis
from .loss import DebiasedFYLoss, NaiveFYLoss
from .optimizer import DebiasedFYEstimator, EstimatorConfig
from .projection import GammaProjection, project_bernstein, project_nonneg
from .ustats import falling_factorial, u_statistics
from .variance import per_od_scores, sandwich_variance

__all__ = [
    "SieveBasis",
    "DebiasedFYLoss",
    "NaiveFYLoss",
    "DebiasedFYEstimator",
    "EstimatorConfig",
    "GammaProjection",
    "project_bernstein",
    "project_nonneg",
    "falling_factorial",
    "u_statistics",
    "per_od_scores",
    "sandwich_variance",
]
