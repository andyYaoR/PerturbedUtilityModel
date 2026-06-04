# Global Convergence Roadmap

This document records the current state of the PURCSolver inner solver and the
staged development path toward a robust, fast, theoretically justified algorithm
for general PURC forward problems.

The target problem is the paper's inner PURC program

```text
min_x F(x; gamma) - v(beta)^T x
subject to A x = b, 0 <= x <= 1,
F(x; gamma) = sum_i ell_i h(x_i; gamma).
```

The dual implementation uses multipliers `lambda`, reduced utilities
`eta = (v + A^T lambda) / ell`, primal recovery `x(lambda) = xi*(eta)`, residual
`r(lambda) = A x(lambda) - b`, and generalized Hessian
`H = A_S diag(1 / (ell_i h''(x_i))) A_S^T` on the strict interior active set.

## Current Findings

Small tests pass:

```text
pytest -q
145 passed
```

Implementation note: `success=True` is now tied to the final feasibility
residual `||Ax-b||_inf < tol`. The LM objective-noise branch may accept a step,
but it no longer certifies convergence by itself. This matters because prototype
continuation runs exposed cases where objective-scale progress became tiny while
the final residual was still above the requested tolerance.

Large cold-start single-OD benchmarks do not currently meet the goal. The
benchmark previously skipped failed OD pairs silently; `dev/bench_chicago.py` now
reports convergence counts and the best residual reached by failed OD pairs.

Corrected Regional baseline, `python dev/bench_chicago.py regional --no-cvxpy`:

```text
perturbation  beta  conv   SSN ms  nit    resid failbest
quadratic      1.0  1/3     861.9  148    1e-11  6.3e-03
quadratic      5.0  1/3    2021.3  347    2e-10  9.9e-03
entropy        1.0  3/3      82.5   19    8e-10        -
entropy        5.0  3/3     286.3   36    7e-10        -
sieve(L=4)     1.0  1/3    1913.6  336    5e-10  1.1e-02
sieve(L=4)     5.0  0/3         -    -        -  1.0e-02
```

Sketch is better but still has a hidden hard-kernel failure:

```text
perturbation  beta  conv
quadratic      1.0  3/3
quadratic      5.0  3/3
entropy        1.0  3/3
entropy        5.0  3/3
sieve(L=4)     1.0  3/3
sieve(L=4)     5.0  2/3
```

The failure mode is concentrated in hard-saturation perturbations with
`h'(0)=0` and negative utilities. At the cold start `lambda=0`, quadratic and
polynomial-sieve instances often recover `x=0`, so the selected active set is
empty and `H=0`. The current Levenberg-Marquardt objective globalization then
grows the active set gradually through node potentials. On failing Regional ODs,
the dual objective keeps decreasing, but stationarity residuals can drop near
`1e-2` and later jump back to order `1`, eventually hitting `max_iter`.

Entropy avoids this because its recovery has full support for finite `eta`, so
the Newton system is informative from the first iteration.

## Proof Gap

The paper's algorithm states a regularized semismooth Newton step with

```text
epsilon_k = O(||r^k||)
Armijo line search on the convex dual objective
```

and cites global convergence for regularized semismooth Newton. The current code
uses a different Levenberg-Marquardt/trust-region acceptance ratio with persisted
damping. This may be useful, but it is not the algorithm described in the paper,
and the observed large-instance behavior shows that objective decrease alone is
not a sufficient practical global phase for hard-saturation cold starts.

A direct prototype of the paper-style Armijo loop did not fix the failing
Regional OD from the same cold start. This suggests the next step should not be a
cosmetic switch from LM back to Armijo; the solver needs a principled global
initialization/continuation phase.

## Development Principles

1. Preserve the exact target problem. Smoothing, barriers, or proximal terms are
   allowed only as continuation/globalization devices with an exact final solve.
2. Every algorithmic parameter must either be fixed by a theorem, derived from
   problem constants, or controlled by an acceptance rule. No per-network hand
   tuning.
3. Each stage must have correctness gates against CVXPY for DCP kernels and
   against the independent scipy dual oracle for polynomial-sieve kernels.
4. Benchmarks must report all failures, not only successful timings.
5. Performance claims must separate iteration count, linear-solve backend, and
   recovery/assembly costs.

## Literature Anchors

The algorithmic direction below is grounded in the following primary references.

- Qi and Sun, ["A nonsmooth version of Newton's method"](https://doi.org/10.1007/BF01581275),
  Mathematical Programming, 1993. This is the local semismooth-Newton foundation:
  generalized Jacobians replace classical derivatives, and semismoothness gives
  the fast local convergence theory.
- Qi, ["Convergence Analysis of Some Algorithms for Solving Nonsmooth
  Equations"](https://doi.org/10.1287/moor.18.1.227), Mathematics of Operations
  Research, 1993. This is a useful warning for our current LM baseline: global
  damping alone is not the same as a complete global convergence proof to a zero
  of the nonsmooth system; hybrid/globalized merit constructions matter.
- Hintermuller, Ito, and Kunisch, ["The Primal-Dual Active Set Strategy as a
  Semismooth Newton Method"](https://doi.org/10.1137/S1052623401383558), SIAM
  Journal on Optimization, 2002. This supports the active-set interpretation of
  the exact PURC SSN step and emphasizes merit functions for global convergence.
- Nesterov and Nemirovskii, ["Interior-Point Polynomial Algorithms in Convex
  Programming"](https://doi.org/10.1137/1.9781611970791), SIAM, 1994/2012. This
  is the central-path/self-concordant barrier reference for convex programming.
- Wright, ["Primal-Dual Interior-Point Methods"](https://books.google.com/books?id=oQdBzXhZeUkC),
  SIAM, 1997. This is the practical primal-dual path-following reference,
  including predictor-corrector ideas and sparse linear algebra.
- Boyd and Vandenberghe, ["Convex Optimization"](https://www.seas.ucla.edu/~vandenbe/cvxbook.html),
  Cambridge University Press, 2004. This gives the standard barrier-method and
  central-path treatment for convex equality-constrained problems.
- Wachter and Biegler, ["On the Implementation of an Interior-Point Filter
  Line-Search Algorithm for Large-Scale Nonlinear
  Programming"](https://doi.org/10.1007/s10107-004-0559-y), Mathematical
  Programming, 2006. This is not the proof template for our convex problem, but
  it is a relevant implementation reference for large-scale primal-dual
  interior-point globalization and restoration logic.

## Staged Plan

### Stage 0: Instrumentation and Honest Baselines

Status: started.

Deliverables:

- Report convergence counts in `dev/bench_chicago.py`.
- Add per-OD diagnostic output for residual history, best residual, active-set
  size, damping, backend route, and final status.
- Add benchmark fixtures for the currently failing Regional ODs so regressions
  are deterministic.
- Promote "all sampled ODs converged" to a benchmark gate before timing medians
  are interpreted.

### Stage 1: Mathematical Invariants and Solver Contracts

Deliverables:

- Write tests that verify the implemented dual signs, KKT complementarity, and
  conjugate values on general polytopes, incidence polytopes, and rank-deficient
  incidence systems.
- Make multiplier normalization explicit for diagnostics, even if the primal is
  gauge-invariant. This keeps dual objective histories interpretable.
- Track best primal-feasibility residual separately from final residual, but do
  not return a best iterate as "success" unless KKT stationarity is certified.

### Stage 2: Theory-Backed Global Phase

Candidate A: specialized primal-dual barrier continuation.

Solve the perturbed box problem with a logarithmic barrier,

```text
min_x F(x; gamma) - v^T x
      - mu sum_i [log(x_i - lo_i) + log(hi_i - x_i)]
subject to A x = b,
```

then decrease `mu` by a duality-gap rule and finish with exact active-set SSN at
`mu=0`. The Newton Schur complement remains
`A diag(1 / (ell_i h''(x_i) + barrier_hess_i)) A^T`, so it reuses the same
Laplacian backend. This is the closest specialized analogue of CVXPY's robust
interior-point behavior, but without generic modeling overhead.

For a fixed `mu > 0`, the per-coordinate recovery for a multiplier `lambda`
becomes a strictly monotone scalar equation on `(lo_i, hi_i)`:

```text
ell_i h'(x_i; gamma) - v_i - (A^T lambda)_i
    - mu / (x_i - lo_i) + mu / (hi_i - x_i) = 0.
```

The derivative is

```text
ell_i h''(x_i; gamma)
    + mu / (x_i - lo_i)^2 + mu / (hi_i - x_i)^2 > 0,
```

so the dual residual is smooth for fixed `mu`, every coordinate has positive
curvature, and the Newton matrix keeps the same `A diag(w) A^T` form with

```text
w_i(mu) = 1 /
  [ell_i h''(x_i; gamma)
   + mu/(x_i-lo_i)^2
   + mu/(hi_i-x_i)^2].
```

This directly addresses the empty-active-set cold start for quadratic and sieve
kernels: with `mu > 0`, there is no zero-curvature active-set selection at the
start. The proof obligation is also cleaner than the current LM baseline:
fixed-`mu` subproblems are smooth, strictly convex barrier subproblems; the
central-path theory controls `mu -> 0`; exact SSN at `mu=0` is only used after
the barrier path has produced a high-quality warm start.

Important feasibility caveat: a log barrier requires a strictly feasible point
with `lo < x < hi` and `Ax = b` on the coordinates being barriered. For strongly
connected PURC networks this is plausible because circulations can make all arcs
interior before the negative-utility optimum removes cycles, but it is not a
free assumption for arbitrary general polytopes. The production algorithm needs
either a Phase-I strict-feasibility routine, a reduced-coordinate barrier over
coordinates that admit interior movement, or a documented fallback when Slater
fails.

Candidate B: proximal dual continuation.

Solve a sequence of strongly convex dual subproblems

```text
min_lambda phi(lambda) + (mu / 2) ||lambda - lambda_ref||^2
```

with semismooth Newton and Armijo, reducing `mu` only after the proximal
stationarity residual is small. The current LM step resembles one iteration of
this idea, but the continuation subproblem is not solved to a controlled
tolerance. A structured proximal path gives a clearer proof obligation.

Prototype evidence: a conservative proximal path is stable on the hard Regional
sieve case and can drive the residual down monotonically along the path, but it is
too slow as a standalone global phase. For OD `6141 -> 6642`, reducing `mu` by
`0.5` from `1` to about `1e-8` required thousands of inner Newton iterations and
still handed exact SSN a residual around `1e-6` to `1e-5`. This makes candidate B
useful as a proof template and diagnostic baseline, but not yet the performance
solution.

Candidate C: KKT-derived graph warm starts for PURC incidence problems.

For hard-saturation kernels, construct deterministic initial potentials and/or a
strictly feasible primal seed from shortest-path marginal costs derived from the
box KKT inequalities. This should be treated as a warm-start accelerator, not as
the only globalization mechanism, and must fall back to the general global phase
for arbitrary polytopes.

Stage 2 should prototype A and B on the fixed failing OD set, then keep the one
with the cleaner proof and best iteration profile. Given the initial proximal
results, candidate A should be prioritized next: a primal-dual barrier path has
full interior curvature from the first iteration, directly addresses the empty
active-set cold start, and still preserves an exact final `mu=0` SSN polish.

### Stage 3: Exact SSN Polishing

Once the global phase reaches the active-set basin, use the exact target problem
with no barrier/proximal term:

```text
(H^k + epsilon_k I) d = -r^k,
epsilon_k = O(||r^k||),
Armijo or certified trust-region globalization.
```

Acceptance must be tied to a theorem for strongly semismooth gradients of convex
dual functions. The implementation should expose enough diagnostics to verify
that `epsilon_k -> 0` and that local convergence is superlinear/quadratic on
stable active sets.

### Stage 4: Backend Speed

Deliverables:

- Support directed parallel arcs in the incidence fast path by summing Newton
  weights over unordered endpoint pairs for `A diag(w) A^T`, while keeping the
  original directed columns for `A^T lambda`, recovery, and output flows.
- Use the native cached PURC assembly path whenever incidence is detected.
- Exploit shared-matrix batches for quadratic and per-system batches for
  nonquadratic kernels.
- Revisit active-subgraph forest routing only after convergence is fixed; it is
  an optimization, not a correctness dependency.

### Stage 5: Acceptance Gates

The solver is not considered complete until these gates pass:

- All existing unit tests pass.
- Fixed failing Regional ODs converge for quadratic and sieve at beta 1 and beta 5.
- DCP kernels match CVXPY within the existing primal tolerance on all benchmarked
  instances.
- Polynomial sieve matches the scipy oracle on small/medium cases and satisfies
  KKT residual gates on large cases where CVXPY is unavailable.
- Single-OD timings beat CVXPY on every DCP benchmark row; batched timings are
  reported separately and should improve further.
