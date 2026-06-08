.. _examples:

Examples
========

This page walks through the runnable scripts in ``examples/``.  Each is
self-contained and prints a short summary; run any of them from the repository
root:

.. code-block:: bash

   python examples/<script_name>.py

Forward solver
--------------

Solving the perturbed-utility equilibrium for a network and an
origin--destination demand.

forward_solve.py
^^^^^^^^^^^^^^^^

**Script:** ``examples/forward_solve.py``

Builds a :class:`~purc.static_purc.PUMProblem` on the SiouxFalls network, preprocesses
the unified :func:`~purc.static_purc.get_solver` once, and solves a single
origin--destination demand and a batch of demands, inspecting the
:class:`~purc.static_purc.PURCResult`.

.. literalinclude:: ../../../../examples/forward_solve.py
   :language: python
   :pyobject: main
   :caption: forward_solve.py -- the high-level solver interface

perturbation_comparison.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Script:** ``examples/perturbation_comparison.py``

Solves the same origin--destination problem under several perturbation kernels
(quadratic, Shannon entropy, modified entropy, polynomial sieve) and reports how
each splits the unit demand between two routes -- a compact picture of how the
kernel shapes choice.

.. literalinclude:: ../../../../examples/perturbation_comparison.py
   :language: python
   :pyobject: main
   :caption: perturbation_comparison.py -- how the kernel shapes route choice

Estimation
----------

Recovering the utility coefficients ``beta`` and the perturbation shape ``gamma``
from observed route-choice frequencies with the debiased Fenchel--Young estimator.

estimate.py
^^^^^^^^^^^

**Script:** ``examples/estimate.py``

The complete estimation workflow on a small synthetic network: simulate trips at a
known parameter with :func:`~purc.static_purc.dgp.simulate_dataset`, fit
:class:`~purc.estimators.debiased_fy.DebiasedFYEstimator`, and report parameter
recovery, sandwich standard errors, and 95% confidence intervals.

.. literalinclude:: ../../../../examples/estimate.py
   :language: python
   :pyobject: main
   :caption: estimate.py -- end-to-end estimation and inference

estimate_siouxfalls.py
^^^^^^^^^^^^^^^^^^^^^^^

**Script:** ``examples/estimate_siouxfalls.py``

The same workflow on the SiouxFalls road network, with a two-dimensional utility
(free-flow time and congestion proneness) and a cubic sieve coefficient.

.. literalinclude:: ../../../../examples/estimate_siouxfalls.py
   :language: python
   :pyobject: main
   :caption: estimate_siouxfalls.py -- estimation on a real network

.. note::

   The lower-level ``purc.laplaciansolve`` SDDM / Laplacian solver, which the
   forward solver calls internally for each Newton system, is documented in the
   :doc:`../api/laplaciansolve` API reference.
