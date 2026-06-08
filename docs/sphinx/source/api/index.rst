API Reference
=============

The complete API reference for PURCSolver, generated from the in-source
Google-style docstrings.

static_purc -- forward solver
-----------------------------

The static PURC model: problems, perturbation kernels, polytope constraints, the
forward solvers, and the data-generating process.

.. toctree::
   :maxdepth: 2

   static_purc

estimators -- debiased Fenchel--Young
-------------------------------------

The estimator interface and the debiased Fenchel--Young estimator with sandwich
inference.

.. toctree::
   :maxdepth: 2

   estimators

laplaciansolve -- Laplacian / SDDM solver
-----------------------------------------

The vendored sparse linear solver behind every forward-solve Newton step.

.. toctree::
   :maxdepth: 2

   laplaciansolve
