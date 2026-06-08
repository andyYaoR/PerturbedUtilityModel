"""
Horse-race: every PURCSolver engine vs CVXPY on the Chicago road networks.

Compares the candidate forward-solve engines

    ssn      -- dual regularized semismooth Newton (sound LM globalization)
    ipm      -- primal-dual interior-point (Mehrotra) + SSN crossover
    barrier  -- barrier-smoothed dual continuation + SSN crossover

against CVXPY (Clarabel) across perturbations (quadratic / entropy / sieve),
utility scales (beta), and networks (Sketch / Regional).  Each engine is run on
the same OD-pairs; we report convergence, Newton iterations, median per-solve
time, KKT residual, and the max primal discrepancy vs CVXPY (where DCP-expressible).

The goal (see memory ``purc-single-algorithm-goal``): one engine that converges
AND beats CVXPY on every cell.  Engines that hit a *provable* precondition (the
IPM on a Legendre-type kernel) are reported as ``precond`` -- not a failure, a
structural skip.

Run:
    python dev/horserace.py sketch
    python dev/horserace.py regional
    python dev/horserace.py sketch --engines ssn,ipm,barrier --no-cvxpy
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
from tntp import load_net

from purc.static_purc import PUMProblem, SSNConfig
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers import get_solver
from purc.static_purc.utils.torch_compat import to_numpy

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data") + os.sep
NETS = {
    "sketch": DATA + "ChicagoSketch_net.tntp",
    "regional": DATA + "ChicagoRegional_net.tntp",
}

# perturbation name -> (constructor, cvxpy elementwise h or None, gamma)
import cvxpy as cp  # noqa: E402

PERTS = {
    "quadratic": (lambda: get_perturbation("quadratic"), lambda x: 0.5 * cp.square(x), np.zeros(0)),
    "entropy": (lambda: get_perturbation("entropy"), lambda x: -cp.entr(x), np.zeros(0)),
    "mod_entropy": (
        lambda: get_perturbation("modified_entropy"),
        lambda x: -cp.entr(1.0 + x) - x,
        np.zeros(0),
    ),
    "sieve": (
        lambda: get_perturbation("polynomial_sieve", gamma=np.array([0.3, 0.1])),
        None,
        np.array([0.3, 0.1]),
    ),
}


def _median_time(fn, repeats=3):
    """Warm up once, then return the median wall-clock of ``repeats`` calls (ms)."""
    fn()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times)


def _feasible_ods(n, n_od, seed=1):
    rng = np.random.default_rng(seed)
    ods = []
    while len(ods) < n_od:
        o, d = int(rng.integers(0, n)), int(rng.integers(0, n))
        if o != d:
            ods.append((o, d))
    return ods


def _make_solver(name):
    cfg = SSNConfig(tol=1e-9, max_iter=400)
    return get_solver(name, config=cfg)


def run(net_key, engines, use_cvxpy, n_od):
    """Run the horse-race for one network and print per-cell solver tables."""
    net = load_net(NETS[net_key])
    A, n, m = net.A, net.n_nodes, net.n_links
    cost = np.maximum(net.length, 1e-3)
    ods = _feasible_ods(n, n_od)
    print(f"\n=== Chicago {net_key} (n={n}, m={m}) | {n_od} OD-pairs ===")

    for pert_name, (make_pert, cvxpy_h, gamma) in PERTS.items():
        for beta in (1.0, 5.0):
            v = -beta * cost
            poly = GeneralPolytope(A, np.zeros(n), ell=cost, validate=False)
            prob = PUMProblem(make_pert(), poly)

            # CVXPY reference (once per cell, first OD) for timing + accuracy.
            cvx_ms, xref_by_od, cvx_fail = [], {}, 0
            if use_cvxpy and cvxpy_h is not None:
                for o, d in ods:
                    bvec = np.zeros(n)
                    bvec[o], bvec[d] = 1.0, -1.0
                    x = cp.Variable(m)
                    pr = cp.Problem(
                        cp.Minimize(cp.sum(cp.multiply(cost, cvxpy_h(x))) - v @ x),
                        [A @ x == bvec, x >= 0, x <= 1],
                    )
                    t = time.perf_counter()
                    try:
                        pr.solve(solver=cp.CLARABEL)
                    except cp.error.SolverError:
                        cvx_fail += 1
                        xref_by_od[(o, d)] = None
                        continue
                    cvx_ms.append((time.perf_counter() - t) * 1e3)
                    xref_by_od[(o, d)] = None if x.value is None else np.asarray(x.value)

            print(f"\n  {pert_name} | beta={beta:.0f}")
            cvx_str = f"{statistics.median(cvx_ms):.1f}ms" if cvx_ms else "-"
            fail_note = f"  ({cvx_fail}/{n_od} FAILED)" if cvx_fail else ""
            print(f"    {'cvxpy':10s}  median={cvx_str}{fail_note}")
            header = f"    {'engine':10s} {'conv':>6s} {'nit':>4s} {'med ms':>8s} {'resid':>8s} {'vs cvxpy':>9s} {'speedup':>8s}"
            print(header)

            for eng in engines:
                try:
                    solver = _make_solver(eng)
                    solver.preprocess(prob)
                except ValueError as exc:
                    # provable precondition (e.g. IPM on a Legendre kernel)
                    reason = "precond" if "essentially smooth" in str(exc) else "skip"
                    print(f"    {eng:10s} {reason:>6s} {'-':>4s} {'-':>8s} {'-':>8s} {'-':>9s} {'-':>8s}")
                    continue

                ms, nits, resids, errs, conv = [], [], [], [], 0
                for o, d in ods:
                    bvec = np.zeros(n)
                    bvec[o], bvec[d] = 1.0, -1.0

                    def _solve(_b=bvec):
                        return solver.solve((v, gamma), b=_b, lam0=np.zeros(n))

                    res = _solve()
                    if not res.success:
                        continue
                    conv += 1
                    ms.append(_median_time(_solve))
                    nits.append(res.nit)
                    resids.append(res.residual)
                    xref = xref_by_od.get((o, d))
                    if xref is not None:
                        errs.append(float(np.abs(to_numpy(res.x).ravel() - xref).max()))

                if not ms:
                    print(f"    {eng:10s} {f'0/{n_od}':>6s} {'-':>4s} {'-':>8s} {'-':>8s} {'-':>9s} {'-':>8s}")
                    continue
                med = statistics.median(ms)
                spd = (statistics.median(cvx_ms) / med) if cvx_ms else float("nan")
                err_str = f"{statistics.median(errs):.1e}" if errs else "-"
                spd_str = f"{spd:.1f}x" if spd == spd else "-"
                print(
                    f"    {eng:10s} {f'{conv}/{n_od}':>6s} {int(statistics.median(nits)):>4d} "
                    f"{med:>8.1f} {statistics.median(resids):>8.0e} {err_str:>9s} {spd_str:>8s}"
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("net", nargs="?", default="sketch", choices=list(NETS))
    ap.add_argument("--engines", default="ssn,ipm,barrier")
    ap.add_argument("--no-cvxpy", action="store_true")
    ap.add_argument("--n-od", type=int, default=3)
    args = ap.parse_args()
    run(args.net, args.engines.split(","), not args.no_cvxpy, args.n_od)
