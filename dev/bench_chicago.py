"""
Comprehensive single-RHS (one OD-pair) benchmark on the Chicago road networks.

Compares the regularized semismooth Newton solver against CVXPY (Clarabel) across
perturbations, utility scales, and perturbation scales on Chicago Sketch
(n=933, m=2950) and Chicago Regional (n=12982, m=39018).

Each row is a configuration; we report the (preprocessed) per-solve time as the
median over a few OD-pairs and timed repeats, the Newton iteration count, the KKT
residual, the CVXPY time (where the perturbation is DCP-expressible), and the
max primal discrepancy SSN-vs-CVXPY.

Run:
    python dev/bench_chicago.py sketch          # Sketch, all perturbations + CVXPY
    python dev/bench_chicago.py regional        # Regional, CVXPY only for quadratic
    python dev/bench_chicago.py sketch --no-cvxpy
"""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import torch

from purcsolver import PUMProblem, SSNConfig
from purcsolver.constraints import GeneralPolytope
from purcsolver.perturbations import get_perturbation
from purcsolver.solvers import RegularizedSSNSolver
from purcsolver.utils.torch_compat import to_numpy
from tntp import load_net

DATA = "/Users/ruiyao/Library/CloudStorage/Dropbox/Technion/Codes/LaplacianSolve/examples/data/"
NETS = {
    "sketch": DATA + "ChicagoSketch_net.tntp",
    "regional": DATA + "ChicagoRegional_net.tntp",
}

# Perturbations to benchmark.  cvxpy_h is None when CVXPY cannot express it.
import cvxpy as cp  # noqa: E402

PERTS = {
    "quadratic": (lambda: get_perturbation("quadratic"), lambda x: 0.5 * cp.square(x), torch.zeros(0)),
    "entropy": (lambda: get_perturbation("entropy"), lambda x: -cp.entr(x), torch.zeros(0)),
    "sieve(L=4)": (
        lambda: get_perturbation("polynomial_sieve", gamma=np.array([0.3, 0.1])),
        None,
        np.array([0.3, 0.1]),
    ),
}


def _median_time(fn, repeats=5):
    """Warm up once, then return the median wall-clock of ``repeats`` calls (ms)."""
    fn()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times)


def _feasible_ods(n, n_od, seed=0):
    rng = np.random.default_rng(seed)
    ods = []
    while len(ods) < n_od:
        o, d = int(rng.integers(0, n)), int(rng.integers(0, n))
        if o != d:
            ods.append((o, d))
    return ods


def run(net_key: str, use_cvxpy: bool) -> None:
    """Run the benchmark for one network and print a results table."""
    net = load_net(NETS[net_key])
    A, n, m = net.A, net.n_nodes, net.n_links
    cost = np.maximum(net.length, 1e-3)
    ods = _feasible_ods(n, 3, seed=1)

    print(f"\n=== Chicago {net_key} (n={n}, m={m}) ===")
    print(
        f"{'perturbation':12s} {'beta':>5s} {'conv':>5s} {'SSN ms':>8s} "
        f"{'nit':>4s} {'resid':>8s} {'failbest':>8s} {'CVXPY ms':>9s} "
        f"{'status':>9s} {'max|dx|':>9s}"
    )

    for pert_name, (make_pert, cvxpy_h, gamma) in PERTS.items():
        for beta in (1.0, 5.0):
            v = -beta * cost
            poly = GeneralPolytope(A, np.zeros(n), ell=cost, validate=False)
            prob = PUMProblem(make_pert(), poly)
            solver = RegularizedSSNSolver(SSNConfig(tol=1e-9, max_iter=400))
            solver.preprocess(prob)

            ssn_ms, nits, resids, errs, cvx_ms, status = [], [], [], [], [], "-"
            fail_best = []
            for o, d in ods:
                b = np.zeros(n)
                b[o] = 1.0
                b[d] = -1.0

                def _solve(_b=b):
                    return solver.solve((v, gamma), b=_b, lam0=np.zeros(n))

                res = _solve()
                if not res.success:
                    if res.residual_history:
                        fail_best.append(min(res.residual_history))
                    else:
                        fail_best.append(res.residual)
                    continue
                ssn_ms.append(_median_time(_solve, repeats=3))
                nits.append(res.nit)
                resids.append(res.residual)

                if use_cvxpy and cvxpy_h is not None:
                    x = cp.Variable(m)
                    pr = cp.Problem(
                        cp.Minimize(cp.sum(cp.multiply(cost, cvxpy_h(x))) - v @ x),
                        [A @ x == b, x >= 0, x <= 1],
                    )
                    t = time.perf_counter()
                    pr.solve(solver=cp.CLARABEL)
                    cvx_ms.append((time.perf_counter() - t) * 1e3)
                    status = pr.status
                    if x.value is not None:
                        errs.append(float(np.abs(to_numpy(res.x) - x.value).max()))

            if not ssn_ms:
                fail_str = f"{min(fail_best):8.1e}" if fail_best else f"{'-':>8s}"
                print(
                    f"{pert_name:12s} {beta:5.1f} {0:2d}/{len(ods):<2d} "
                    f"{'-':>8s} {'-':>4s} {'-':>8s} {fail_str} "
                    f"{'-':>9s} {'-':>9s} {'-':>9s}"
                )
                continue
            cvx_str = f"{statistics.median(cvx_ms):9.1f}" if cvx_ms else f"{'-':>9s}"
            err_str = f"{statistics.median(errs):9.1e}" if errs else f"{'-':>9s}"
            fail_str = f"{min(fail_best):8.1e}" if fail_best else f"{'-':>8s}"
            print(
                f"{pert_name:12s} {beta:5.1f} {len(ssn_ms):2d}/{len(ods):<2d} "
                f"{statistics.median(ssn_ms):8.1f} {int(statistics.median(nits)):4d} "
                f"{statistics.median(resids):8.0e} {fail_str} {cvx_str} "
                f"{status:>9s} {err_str}"
            )


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "sketch"
    no_cvxpy = "--no-cvxpy" in sys.argv
    run(key, use_cvxpy=not no_cvxpy)
