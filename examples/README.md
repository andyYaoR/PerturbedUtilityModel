# Examples

Self-contained, runnable scripts that demonstrate the main features of PURCSolver.
Each script prints a short summary and can be run from the repository root, for
example:

```bash
python examples/estimate.py
```

## Estimation

Recover the utility coefficients `beta` and the perturbation shape `gamma` from
observed route-choice frequencies with the debiased Fenchel–Young estimator.

| Example | Description |
| --- | --- |
| [`estimate.py`](estimate.py) | The complete estimation workflow on a small synthetic network: simulate trips at a known parameter, fit the estimator, and report parameter recovery, sandwich standard errors, and 95% confidence intervals. Deterministic and fast. |
| [`estimate_siouxfalls.py`](estimate_siouxfalls.py) | The same workflow on the SiouxFalls road network with a two-dimensional utility (free-flow time and congestion proneness) and a cubic sieve coefficient. |

## Forward solver

Solve the perturbed-utility equilibrium `x*(theta) = argmin F(x; gamma) - v(beta)^T x`.

| Example | Description |
| --- | --- |
| [`forward_solve.py`](forward_solve.py) | The high-level solver interface on SiouxFalls: build a problem, preprocess once, and solve single and batched origin–destination demands. |
| [`perturbation_comparison.py`](perturbation_comparison.py) | How the choice of perturbation kernel (quadratic, Shannon entropy, modified entropy, polynomial sieve) shapes the route split. |

## Supporting files

- [`tntp.py`](tntp.py) — a shared loader for TNTP `_net` files, used by the forward-solver and SiouxFalls examples (not run directly).
- `data/` — bundled networks: `SiouxFalls_net.tntp`, `ChicagoSketch_net.tntp`, `ChicagoRegional_net.tntp`.
```
