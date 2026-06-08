.. PURCSolver documentation master file

Welcome to PURCSolver's documentation!
======================================

**PURCSolver** solves perturbed-utility route-choice models with general polytope
constraints, and estimates them from data with a debiased Fenchel--Young estimator.
The strictly-convex separable program is solved on its convex dual, where each
Newton step reduces to a weighted-graph-Laplacian (SDDM) linear solve handled by the
vendored, GIL-released ``LaplacianSolve`` backend.

The package is **torch-native** (CPU ``float64`` by default, written
device-agnostically) and accepts NumPy/SciPy inputs bridged zero-copy on CPU.  It is
organized as three layers that share a common forward-solve core:

- :mod:`purc.static_purc` -- the static PURC model and its forward solvers
  (the primal--dual interior-point method and the dual semismooth Newton method),
  the perturbation kernels, polytope constraints, and the data-generating process.
- :mod:`purc.estimators` -- estimators built on the forward solver; the debiased
  Fenchel--Young estimator (:mod:`purc.estimators.debiased_fy`) recovers the utility
  and perturbation parameters from observed choices.
- :mod:`purc.laplaciansolve` -- the vendored Laplacian / SDDM linear solver behind
  every forward-solve Newton step.

Quick start
-----------

.. code-block:: bash

   pip install --no-build-isolation -e .
   python examples/estimate.py

.. code-block:: python

   import numpy as np
   import scipy.sparse as sp
   from purc.static_purc import PUMProblem, ForwardSolverConfig
   from purc.static_purc.constraints import GeneralPolytope
   from purc.static_purc.perturbations import get_perturbation
   from purc.static_purc.solvers import get_solver

   A = sp.csr_matrix(np.ones((1, 5)))            # sum(x) = 1
   poly = GeneralPolytope(A, b=np.array([1.0]), lo=0.0, hi=1.0)
   prob = PUMProblem(get_perturbation("modified_entropy"), poly)

   solver = get_solver("ipm", config=ForwardSolverConfig())   # primal-dual interior point
   solver.preprocess(prob)
   # theta = (beta, gamma); torch or numpy inputs are both accepted.
   res = solver.solve((np.array([0.1, -0.4, 0.7, 0.2, -0.1]), np.zeros(0)))
   print(res.x, res.success, res.nit)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
