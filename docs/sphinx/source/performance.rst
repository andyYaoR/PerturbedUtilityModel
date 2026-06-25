Performance
===========

PURCSolver solves the forward problem on its convex dual, where each Newton step is a
symmetric diagonally dominant (SDDM) system -- a weighted graph Laplacian on
networks -- factorized directly by CHOLMOD.  The solver is built once with
:meth:`preprocess` and reused across origin-destination pairs (warm starts and a
cached symbolic factorization), the way an assignment or estimation loop uses it.

The benchmark in ``benchmarks/forward_vs_cvxpy.py`` times this against the *same*
convex program solved by CVXPY's Clarabel, on the TNTP road networks bundled in
``examples/data``.  It uses the modified-entropy kernel so the program is
DCP-expressible and the comparison is apples-to-apples, and reports the maximum
primal discrepancy as a correctness check.

.. code-block:: bash

   pip install --no-build-isolation -e ".[oracle]"   # CVXPY, for the comparison
   python benchmarks/forward_vs_cvxpy.py

Representative results (median per-solve wall time; absolute numbers vary with
hardware and BLAS, but the trend is stable):

.. list-table::
   :header-rows: 1
   :widths: 24 10 10 14 18 12

   * - Network
     - Nodes
     - Links
     - IPM (ms)
     - CVXPY/Clarabel (ms)
     - Speedup
   * - SiouxFalls
     - 24
     - 76
     - 2.5
     - 3.8
     - 1.5x
   * - ChicagoSketch
     - 933
     - 2950
     - 8.4
     - 88
     - 10.5x
   * - ChicagoRegional
     - 12979
     - 39018
     - 279
     - 1620
     - 5.8x

The primal solutions agree with CVXPY to ``~1e-6`` (reported as ``max|Δx|``).
PURCSolver is consistently several-fold faster on the medium and large networks --
the direct Laplacian / CHOLMOD factorization and warm-started reuse outpace a
general-purpose conic solver -- though the exact ratio depends on a network's size
and sparsity rather than growing monotonically with size.

Batched throughput
------------------

For an assignment or estimation sweep over many origin-destination pairs, the solver
exposes :meth:`solve_batch`, which solves the whole set in one GIL-released native
call over the shared, preprocessed problem -- not a Python loop.  On ChicagoSketch,
one ``solve_batch`` over **5000** OD pairs completes in about 39 s (~128 OD/s);
solving the same 5000 pairs independently with CVXPY would take ~450 s (~11x), and
the primal solutions agree to ``~1e-7``.

.. code-block:: bash

   python benchmarks/forward_vs_cvxpy.py --networks ChicagoSketch --batch-od 5000
