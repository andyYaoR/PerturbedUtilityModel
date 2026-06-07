"""
PURC: perturbed utility route-choice models, solvers, and estimators.

This umbrella package groups three layers that share a common forward-solve core:

* :mod:`purc.static_purc` -- the static PURC model and its forward solvers
  (the primal--dual interior-point method and the dual semismooth Newton method),
  the perturbation kernels, polytope constraints, and the data-generating process.
* :mod:`purc.estimators` -- estimators built on top of the forward solver; the
  debiased Fenchel--Young estimator (:mod:`purc.estimators.debiased_fy`) is the
  first of several, all behind a common :class:`~purc.estimators.base.Estimator`
  interface.
* ``purc.dynamic_purc`` -- reserved for the dynamic model (future work); the
  estimators layer is deliberately model-agnostic so it can serve both.

The submodules are imported lazily so that ``import purc`` stays cheap (it does
not force the native core, the optional cvxpy oracle, or torch to load eagerly
beyond what the submodules themselves require).
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__", "static_purc", "estimators"]


def __getattr__(name: str):
    """
    Lazily expose the umbrella's subpackages.

    Args:
        name: Attribute name being accessed.

    Returns:
        The requested subpackage module.

    Raises:
        AttributeError: If *name* is not a public subpackage.

    """
    if name in ("static_purc", "estimators"):
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
