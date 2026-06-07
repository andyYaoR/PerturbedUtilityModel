"""
Estimators for PURC models, behind a common :class:`~purc.estimators.base.Estimator`.

An estimator consumes a :class:`~purc.estimators.base.SimulatedData` bundle (link
counts / empirical frequencies per OD pair) and a forward solver, and returns an
:class:`~purc.estimators.base.EstimationResult`.  The debiased Fenchel--Young
estimator (:mod:`purc.estimators.debiased_fy`) is the first; the layer is
deliberately model-agnostic so it can serve both the static and (future) dynamic
PURC models.
"""

from __future__ import annotations

from .base import EstimationResult, Estimator, SimulatedData

__all__ = ["Estimator", "EstimationResult", "SimulatedData"]
