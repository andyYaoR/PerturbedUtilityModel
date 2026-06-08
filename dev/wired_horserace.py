"""
Horse-race through the production estimator interface: ``method="newton"`` (the
projected damped Newton with the FD Hessian) vs ``method="tr_bfgs"`` (the wired
trust-region BFGS).  Same problem, data, projection, and start; we count inner
batched IPM solves (the cost unit) and wall time.

Run:
    python dev/wired_horserace.py
"""

from __future__ import annotations

import time

import numpy as np
from bb_horserace import CountingSolver, _solver
from estim_dgp import CATALOG, build_network, make_ods, make_problem

from purc.estimators.debiased_fy import DebiasedFYEstimator, EstimatorConfig, GammaProjection
from purc.static_purc.dgp import simulate_dataset
from purc.static_purc.utils.torch_compat import to_numpy

CASES = [
    dict(name="well/nonneg", th0="pos3", net="synth-20", ns=2, ps=3, B=80, D=200, od=7000, sd=2000, proj="nonneg"),
    dict(name="well/bernstein", th0="pos3", net="synth-20", ns=2, ps=3, B=80, D=200, od=7000, sd=2000, proj="bernstein"),
    dict(name="ill/nonneg", th0="pos345", net="synth-30", ns=0, ps=1, B=120, D=500, od=17000, sd=5000, proj="nonneg"),
]


def _fit(prob, th0, data, method, proj):
    s = CountingSolver(_solver())
    s.preprocess(prob)
    cfg = EstimatorConfig(proj=GammaProjection(proj), method=method, max_iter=300, tol_grad=1e-6)
    t0 = time.perf_counter()
    r = DebiasedFYEstimator(prob, s, th0.L, cfg).fit(data)
    return r, s.n_solve_batch, (time.perf_counter() - t0) * 1e3


def main():
    print(f"\n{'case':14s} {'method':8s} {'conv':>5s} {'outer':>5s} {'solves':>7s} {'ms':>8s} "
          f"{'Q':>13s} {'errγ':>8s} {'speedup':>8s}")
    for c in CASES:
        th0 = CATALOG[c["th0"]]
        inc, attrs = build_network(c["net"], c["ns"])
        prob = make_problem(inc, th0, seed=c["ps"], attrs=attrs)
        ods = make_ods(inc.shape[0], c["B"], c["D"], c["od"])
        base = _solver()
        base.preprocess(prob)
        data = simulate_dataset(prob, base, (th0.beta, th0.gamma), ods, np.random.default_rng(c["sd"]))
        rows = {}
        for method in ("newton", "tr_bfgs"):
            r, solves, ms = _fit(prob, th0, data, method, c["proj"])
            rows[method] = (r, solves, ms)
        base_solves = rows["newton"][1]
        for method in ("newton", "tr_bfgs"):
            r, solves, ms = rows[method]
            errg = float(np.abs(to_numpy(r.gamma_hat) - th0.gamma).max())
            spd = f"{base_solves / solves:.1f}x" if method == "tr_bfgs" else "-"
            print(f"{c['name']:14s} {method:8s} {str(r.converged):>5s} {r.n_outer:>5d} {solves:>7d} "
                  f"{ms:>8.0f} {r.objective:>13.6e} {errg:>8.1e} {spd:>8s}")


if __name__ == "__main__":
    main()
