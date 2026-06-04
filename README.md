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

**M0 (scaffold + native toolchain)** — repo, build system, registries, abstract
interfaces, core dataclasses, and a GIL-released native smoke kernel that proves
the end-to-end build path. Concrete perturbations, constraints, and the SSN
solver land in M1+ (see the staged plan).

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
