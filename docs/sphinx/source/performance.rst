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

The primal solutions agree with CVXPY to ``~1e-6`` (reported as ``max|Δx|``).  The
speedup grows with problem size: the direct Laplacian / CHOLMOD factorization and
warm-started reuse pay off most on larger, sparser networks.
