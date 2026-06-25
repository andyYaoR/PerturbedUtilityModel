# PURCSolver

[![CI](https://github.com/andyYaoR/PerturbedUtilityModel/actions/workflows/ci.yml/badge.svg)](https://github.com/andyYaoR/PerturbedUtilityModel/actions/workflows/ci.yml)
[![docs](https://github.com/andyYaoR/PerturbedUtilityModel/actions/workflows/docs.yml/badge.svg)](https://andyYaoR.github.io/PerturbedUtilityModel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Perturbed utility models: estimation, inference, and prediction.**

PURCSolver solves the perturbed-utility choice model over a constraint polytope — and estimates its
utility and perturbation parameters from observed choices. 

The package is **torch-native** (CPU `float64` by default, written
device-agnostically) and accepts NumPy/SciPy inputs.

## Features

- **Forward solvers** — a primal–dual interior-point method (IPM) and a dual
  semismooth Newton method, behind one `get_solver` entry point that selects the
  regime from a provable property of the kernel.
- **Multiple perturbation kernels** — quadratic, Shannon / logit entropy, modified
  entropy, and a flexible polynomial sieve (with a Bernstein convexity
  certificate), via a simple registry.
- **General polytope constraints** — arbitrary sparse `A x = b, l ≤ x ≤ u`.
- **Debiased Fenchel–Young estimator** — recovers the utility coefficients and the
  perturbation shape from observed choice frequencies, with sandwich standard
  errors.

## The problem

For parameters `θ = (β, γ)`, solve the strictly-convex program

```
min_x  F(x; γ) − v(β)ᵀ x      s.t.   A x = b,   l ≤ x ≤ u
F(x; γ) = Σ_i ℓ_i h(x_i; γ)
```

on its convex dual. One multiplier `λ` per equality row gives a closed-form
per-coordinate primal recovery `x̂_i(λ) = ξ*(η_i; γ)`, and each Newton step solves

```
(A_S diag(D) A_Sᵀ + ε_k I) Δλ = −r,   r = b − A x̂(λ)
```

where the matrix is SPD — a **weighted graph Laplacian** when `A` is a node-arc
incidence matrix — and can be solved efficiently at scale.

## Quick start

Install (see [Installation](#installation)), then solve a forward problem with the
interior-point solver:

```python
import numpy as np, scipy.sparse as sp
from purc.static_purc import PUMProblem, ForwardSolverConfig, get_perturbation, get_solver
from purc.static_purc.constraints import GeneralPolytope

A = sp.csr_matrix(np.ones((1, 5)))                 # sum(x) = 1
poly = GeneralPolytope(A, b=np.array([1.0]), lo=0.0, hi=1.0)
prob = PUMProblem(get_perturbation("modified_entropy"), poly)

solver = get_solver("ipm", config=ForwardSolverConfig())   # primal–dual interior point
solver.preprocess(prob)
# theta = (beta, gamma); torch or numpy inputs both accepted (bridged zero-copy)
res = solver.solve((np.array([0.1, -0.4, 0.7, 0.2, -0.1]), np.zeros(0)))
print(res.x, res.success, res.nit)
```

Runnable examples — forward solve, perturbation comparison, and end-to-end estimation
with standard errors — are in [`examples/`](examples/).

## Installation

PURCSolver builds two native extensions (the PURC core and the vendored
LaplacianSolve backend) with scikit-build-core + nanobind.

**SuiteSparse / CHOLMOD is a required dependency** — the direct factorization behind
the batched Newton solve over OD-pairs. conda-forge is the most reliable way to get
it on all platforms, and the only simple one on Windows:

```bash
conda env create -f environment.yml        # SuiteSparse + toolchain (Linux/macOS/Windows)
conda activate purc
pip install --no-build-isolation -e ".[dev]"
```

The full per-platform walkthrough (including an HPC `-march=native` note)
is in the [documentation](https://andyYaoR.github.io/PerturbedUtilityModel/).

## Performance

The forward solve factorizes the dual Newton (SDDM) system directly with CHOLMOD and
reuses it across origin–destination pairs. On the bundled TNTP networks it matches
CVXPY/Clarabel to ~1e-6 while running several-fold faster on the larger networks:

| Network | Nodes | Links | IPM | CVXPY/Clarabel | Speedup |
| --- | --- | --- | --- | --- | --- |
| SiouxFalls | 24 | 76 | 2.5 ms | 3.8 ms | 1.5× |
| ChicagoSketch | 933 | 2950 | 8.4 ms | 88 ms | 10.5× |
| ChicagoRegional | 12979 | 39018 | 279 ms | 1620 ms | 5.8× |

Reproduce with `python benchmarks/forward_vs_cvxpy.py` (needs the CVXPY oracle:
`pip install --no-build-isolation -e ".[oracle]"`); see the
[Performance docs](https://andyYaoR.github.io/PerturbedUtilityModel/performance.html)
for the methodology.

## Documentation

Full documentation is at **https://andyYaoR.github.io/PerturbedUtilityModel/**.


## License

MIT — see [LICENSE](LICENSE).
