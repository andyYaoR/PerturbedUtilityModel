Getting Started
===============

Installation
------------

PURCSolver builds two native extensions with scikit-build-core and nanobind: the
PURC forward-solver core and the vendored ``LaplacianSolve`` (Laplacian / SDDM)
backend.  A single editable install builds both against your environment.  The full
walkthrough lives in the top-level ``README.md``.

System dependencies
^^^^^^^^^^^^^^^^^^^^

A C++17 toolchain and CMake are required.  CHOLMOD / SuiteSparse and CUDA are
**optional** and auto-detected -- without either, the package falls back to the
bundled ``approxChol`` solver.

.. code-block:: bash

   # macOS
   xcode-select --install
   brew install cmake ninja libomp suite-sparse

   # Ubuntu / Debian / WSL2
   sudo apt install -y build-essential cmake ninja-build pkg-config \
     python3-dev libomp-dev libsuitesparse-dev

On **Windows**, install the Visual Studio Build Tools with the "Desktop development
with C++" workload (the MSVC compiler); ``cmake`` and ``ninja`` are installed by the
build step below.  The optional CHOLMOD / SuiteSparse and CUDA backends are not
required -- the build uses the bundled ``approxChol`` solver when they are absent.

Install
^^^^^^^

The LaplacianSolve backend is vendored in-tree (``purc.laplaciansolve``), so a single
editable install builds everything -- the PURC native core and the Laplacian / SDDM
solver.  Because ``--no-build-isolation`` makes the native cores build against the
PyTorch already in your environment, install the build prerequisites (PyTorch
included) first:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip

   # build prerequisites + a CPU build of PyTorch
   pip install "scikit-build-core>=0.10.0" "nanobind>=2.1.0" cmake ninja torch

   # build and install the package (editable)
   pip install --no-build-isolation -e .

For the development extras (tests, linters, and the CVXPY oracle) use
``pip install --no-build-isolation -e ".[dev]"``.  This is exactly the flow run by
the macOS / Windows / Linux CI.

macOS OpenMP
^^^^^^^^^^^^

PyTorch and the vendored solver each ship an OpenMP runtime.  On macOS, set the
duplicate-library flag before importing the package (add it to ``~/.zshrc`` to make
it persistent):

.. code-block:: bash

   export KMP_DUPLICATE_LIB_OK=TRUE

CPU portability (HPC clusters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The native kernels are compiled with ``-march=native`` (when the compiler supports
it), which targets the exact instruction set of the **build** machine.  On a
heterogeneous cluster -- e.g. a login node with AVX-512 and compute nodes with only
AVX2 -- importing the package on a compute node can then raise ``Illegal
instruction`` (SIGILL).  Build on (or on a node matching) the hardware that will run
the code, for example inside an ``srun`` / ``salloc`` session.  Laptops and single
workstations are unaffected.

Basic Usage
-----------

Build a problem from a perturbation kernel and a polytope, preprocess a solver
once, then solve.  The unified :func:`~purc.static_purc.get_solver` entry point
selects its numerical regime from a provable property of the kernel, so no method
or tuning parameter has to be chosen by hand; the same preprocessed solver is
reused across many right-hand sides.

.. code-block:: python

   import numpy as np
   import scipy.sparse as sp
   from purc.static_purc import PUMProblem, ForwardSolverConfig
   from purc.static_purc.constraints import GeneralPolytope
   from purc.static_purc.perturbations import get_perturbation
   from purc.static_purc.solvers import get_solver

   A = sp.csr_matrix(np.ones((1, 5)))            # sum(x) = 1
   poly = GeneralPolytope(A, b=np.array([1.0]), lo=0.0, hi=1.0)
   prob = PUMProblem(get_perturbation("entropy"), poly)

   solver = get_solver("auto", config=ForwardSolverConfig())
   solver.preprocess(prob)
   # theta = (beta, gamma); torch or numpy inputs are both accepted.
   res = solver.solve((np.array([0.1, -0.4, 0.7, 0.2, -0.1]), np.zeros(0)))
   print(res.x, res.success, res.nit)   # res.x is a torch.Tensor

To **estimate** a model from observed route choices, use
:class:`~purc.estimators.debiased_fy.DebiasedFYEstimator`; the
:doc:`examples/index` page walks through a complete fit with standard errors.

Examples
--------

See the :doc:`examples/index` section for runnable scripts covering the forward
solver, the perturbation kernels, and end-to-end estimation with inference.

Running Tests
-------------

.. code-block:: bash

   # macOS:
   export KMP_DUPLICATE_LIB_OK=TRUE
   pytest -q
