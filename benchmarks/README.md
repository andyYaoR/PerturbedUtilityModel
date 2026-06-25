# Benchmarks

Reproducible speed benchmarks for PURCSolver.

## `forward_vs_cvxpy.py`

Times the perturbed-utility **forward solve** (the interior-point method, IPM)
against the *same* convex program solved by **CVXPY / Clarabel**, on the TNTP road
networks bundled in [`../examples/data`](../examples/data). It uses the
modified-entropy kernel — so the program is DCP-expressible and the comparison is
apples-to-apples — and reports the median per-solve wall time, the speedup, and the
maximum primal discrepancy (`max|Δx|`) as a correctness check.

### Install

The benchmark needs the package plus the CVXPY oracle:

```bash
pip install --no-build-isolation -e ".[oracle]"
```

Without CVXPY the script still runs and reports the IPM timings only.

### Run

```bash
python benchmarks/forward_vs_cvxpy.py                 # SiouxFalls + ChicagoSketch
make bench                                            # the same default run

# all three bundled networks (Clarabel is slow on ChicagoRegional, so keep n-od small):
python benchmarks/forward_vs_cvxpy.py \
    --networks SiouxFalls,ChicagoSketch,ChicagoRegional --n-od 2
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--networks` | `SiouxFalls,ChicagoSketch` | comma-separated TNTP names found in `examples/data` |
| `--n-od` | `4` | timed origin–destination pairs per network (plus one untimed warm-up) |
| `--seed` | `1` | RNG seed for OD-pair sampling |

> On macOS, set `KMP_DUPLICATE_LIB_OK=TRUE` first (PyTorch and CHOLMOD each ship a
> libomp).

### Output

```
network           nodes  links    IPM ms  CVXPY ms  speedup   max|Δx|
SiouxFalls           24     76      2.54       3.8     1.5x   6.0e-06
ChicagoSketch       933   2950      8.40      88.2    10.5x   5.8e-08
ChicagoRegional   12979  39018    279.34    1618.4     5.8x   1.5e-07
```

- **IPM ms / CVXPY ms** — median per-solve wall time. The first solve per network is
  an untimed warm-up; the solver is built once with `preprocess` and reused across OD
  pairs (warm starts + a cached factorization), the way an assignment or estimation
  loop uses it.
- **speedup** — `CVXPY ms / IPM ms`.
- **max|Δx|** — largest difference between the IPM and CVXPY primal solutions; it
  should be at solver tolerance (`~1e-6`), confirming the two solve the same program.

Absolute times depend on hardware and BLAS; the relative trend is stable. See the
[Performance page](https://andyYaoR.github.io/PerturbedUtilityModel/performance.html)
of the documentation for discussion.
