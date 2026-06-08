laplaciansolve -- Laplacian / SDDM Solver
=========================================

The vendored sparse linear solver that handles each forward-solve Newton system (an
SDDM / weighted-graph-Laplacian solve).  CHOLMOD / SuiteSparse and CUDA are optional,
auto-detected backends; absent both, the bundled ``approxChol`` iterative solver is
used.

Solver configuration
--------------------

.. autoclass:: purc.laplaciansolve.SolverConfig
   :members:
   :show-inheritance:

Solvers
-------

.. autoclass:: purc.laplaciansolve.SDDMSolver
   :members:
   :show-inheritance:

.. autoclass:: purc.laplaciansolve.BatchedSDDMSolver
   :members:
   :show-inheritance:

.. autoclass:: purc.laplaciansolve.LaplacianSolver
   :members:
   :show-inheritance:

.. autoclass:: purc.laplaciansolve.PURCLaplacianSolver
   :members:
   :show-inheritance:

Functional interface
--------------------

.. autofunction:: purc.laplaciansolve.solve
