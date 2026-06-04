# PURCSolver

**Perturbed utility models with general polytope constraints**, solved by a
**regularized semismooth Newton (SSN)** method. The per-iteration linear solve is
delegated to the optimized, GIL-released
[LaplacianSolve](../LaplacianSolve) backend.

PURCSolver is a structured, modular companion to the
[PUM](../PUM) package (same engineering conventions, **without** the equilibrium
layer). It targets the model class of *"A semi-nonparametric perturbed utility
model"*: independently swappable **perturbations**, **constraint geometries**, and
**solvers**, with native C++ hot paths and comprehensive correctness/performance
testing against CVXPY.

The package is **torch-native**: tensors are the primary data type for the public
API and all solver math (CPU `float64` by default, written device-agnostically).
NumPy/SciPy inputs are accepted and bridged **zero-copy** on CPU
(`torch.from_numpy` / `tensor.numpy()` share memory), and the few SciPy/CVXPY
dependencies (CSC pattern build, oracle, feasibility) sit behind the same bridge.

## The problem

For parameters `θ = (β, γ)`, solve the strictly-convex separable program

```
min_x  F(x; γ) − v(β)ᵀ x      s.t.   A x = b,   l ≤ x ≤ u
F(x; γ) = Σ_i ℓ_i h(x_i; γ)
```

on its convex dual. One multiplier `λ` per equality row gives a closed-form
per-coordinate primal recovery `x̂_i(λ) = ξ*(η_i; γ)`, and each Newton step solves

```
(A_S diag(D) A_Sᵀ + ε_k I) Δλ = −r,   r = b − A x̂(λ)
```

where the matrix is SPD (a **weighted graph Laplacian** when `A` is a node-arc
incidence matrix) and is handed to LaplacianSolve. Strong semismoothness gives
local Q-quadratic convergence; an Armijo line search on the dual globalizes it.

## Architecture

| Package | Role |
| --- | --- |
| `perturbations/` | Registry of separable kernels `h` (quadratic, entropy, modified entropy, Tsallis, polynomial sieve) + a symbolic compiler that auto-derives a closed-form inverse of `h'(ξ)=η` when it exists, else an optimized bracketed-Halley root-find. |
| `constraints/` | Registry of polytopes: `GeneralPolytope` (arbitrary sparse `A`) and `IncidencePolytope` (network fast path). Multiplier gauge-fixing for rank-deficient `A`. |
| `solvers/` | The `RegularizedSSNSolver` and the `ForwardSolver` interface (build-once `preprocess`, cheap repeatable `solve` — estimation-aware: warm starts + batched OD-pairs). |
| `backends/` | Routes the fixed-pattern Newton system to the right LaplacianSolve solver, built once and solved many times. |
| `oracle/` | CVXPY and independent scipy reference solvers for correctness tests. |
| `native/` (`src/purcsolver/`) | nanobind C++ kernels for the hot path — vectorized `ξ*(η)`, fused weight/CSC assembly, and a full GIL-released SSN step. A pure-numpy fallback is kept for parity testing. |

## Status

- **v0.0.0 (M0)** — scaffold, build system, registries, abstract interfaces, core
  dataclasses, GIL-released native smoke kernel.
- **v0.1.0** — forward solver on **fully general** polytopes (rank-deficient `A`
  via the `ε`-regularization — no gauge module needed). Closed-form perturbations
  (quadratic, Shannon/binary entropy, modified entropy); `GeneralPolytope`;
  `RegularizedSSNSolver` driving LaplacianSolve's CHOLMOD SDDM solver via the
  fixed-pattern `CSCAssembler`; CVXPY + independent scipy oracles.
- **v0.2.0 (current)** — the **polynomial sieve** (the paper's semi-nonparametric
  kernel) with Bernstein convexity certificate `Mγ ≥ −1`, vectorized
  safeguarded-Newton recovery, and the **symbolic compiler**: from a SymPy `h` it
  auto-derives `h'`,`h''`, certifies convexity, and builds a closed-form inverse of
  `h'(ξ)=η` when one exists (else falls back to the root-find). Sieve solves match
  the independent scipy dual oracle to `~1e-15`; closed-form vs root-find agree to
  `~1e-9`.

Next: native compiled recovery kernels — both a parameterized C++ kernel and the
runtime-codegen path, cross-checked for performance (v0.3.0). See the staged plan.

### Quick example

```python
import torch, numpy as np, scipy.sparse as sp
from purcsolver import PUMProblem, SSNConfig
from purcsolver.constraints import GeneralPolytope
from purcsolver.perturbations import get_perturbation
from purcsolver.solvers import RegularizedSSNSolver

A = sp.csr_matrix(np.ones((1, 5)))          # sum(x) = 1
poly = GeneralPolytope(A, b=torch.tensor([1.0]), lo=0.0, hi=1.0)
prob = PUMProblem(get_perturbation("entropy"), poly)

solver = RegularizedSSNSolver(SSNConfig())
solver.preprocess(prob)
# theta = (beta, gamma); torch or numpy inputs both accepted (bridged zero-copy)
res = solver.solve((torch.tensor([0.1, -0.4, 0.7, 0.2, -0.1]), torch.zeros(0)))
print(res.x, res.success, res.nit)   # res.x is a torch.Tensor; res.x.numpy() is zero-copy
```

## Install (development)

LaplacianSolve is a local sibling dependency; install it first, then PURCSolver
editable so the native core builds against your environment:

```bash
pip install --no-build-isolation -e ../LaplacianSolve   # if not already installed
pip install --no-build-isolation -e .
```

## Development

```bash
make build        # editable install with the native core
make test         # pytest
make test-native  # native parity tests
make lint         # ruff + pydocstyle + darglint
make format       # ruff format + autofix
```

Pre-commit hooks (ruff, pydocstyle, darglint, directory-boundary check) mirror the
PUM/LaplacianSolve setup:

```bash
pre-commit install
```
