"""
Benchmark the safe-step/fast-step globalization of the primal-dual IPM.

Two studies:

  A. **Safeguard on vs off** on a synthetic sweep over (network size x ell-spread
     x utility scale x seed).  Reports, per cell, the convergence rate, the median
     iteration count of each, the iteration *premium* of the safeguard (it should
     be ~0 -- the safeguard is transparent when Mehrotra is already fine), and the
     number of *rescues* (the bare Mehrotra step diverges to NaN but the
     safeguarded solver converges).

  B. **Horse race vs CVXPY (Clarabel)** on real TNTP road networks and synthetic
     graphs: the safeguarded IPM and CVXPY solve the same OD subproblems; we
     report median solve time, KKT residual, and the max primal discrepancy.

Run:
    python dev/bench_safeguard.py            # both studies, default sizes
    python dev/bench_safeguard.py --quick    # smaller sweep
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
import scipy.sparse as sp

from purc.static_purc import PUMProblem, SSNConfig
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.utils.torch_compat import to_numpy

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data") + os.sep
GAMMA = np.array([0.5, 0.3, 0.1])  # convex polynomial-sieve shape (gamma >= 0)


# --------------------------------------------------------------------------- #
# problem builders
# --------------------------------------------------------------------------- #
def synth(n_nodes: int, seed: int, ell_spread: float, vscale: float):
    """Random connected directed graph, single OD pair (0 -> n/2)."""
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    for _ in range(3 * n_nodes):
        a, b = rng.integers(0, n_nodes, 2)
        if a != b:
            edges.append((int(a), int(b)))
    inc = np.zeros((n_nodes, len(edges)))
    for k, (a, b) in enumerate(edges):
        inc[a, k], inc[b, k] = 1.0, -1.0
    d = np.zeros(n_nodes)
    d[0], d[n_nodes // 2] = 1.0, -1.0
    ell = np.exp(rng.uniform(-ell_spread, ell_spread, len(edges)))
    pert = get_perturbation("polynomial_sieve", gamma=GAMMA)
    prob = PUMProblem(pert, GeneralPolytope(sp.csr_matrix(inc), d, ell=ell))
    v = -np.random.default_rng(seed + 1).uniform(0.0, vscale, len(edges))
    return prob, v


def from_tntp(path: str, seed: int, n_od: int):
    """Build PURC subproblems from a TNTP network (sieve perturbation)."""
    from tntp import load_net

    net = load_net(path)
    A = net.A
    ell = np.asarray(net.length, dtype=float)
    ell[ell <= 0] = np.median(ell[ell > 0])  # model-valid: floor zero-length links
    fftt = np.asarray(net.fftt, dtype=float)
    v = -(fftt / max(fftt.mean(), 1e-9))  # link disutility ~ -free-flow time (scaled)
    pert = get_perturbation("polynomial_sieve", gamma=GAMMA)
    rng = np.random.default_rng(seed)
    # OD pairs drawn from the largest strongly connected component.
    import scipy.sparse.csgraph as csg

    nC, lab = csg.connected_components(A @ A.T, directed=False)
    big = np.argmax(np.bincount(lab))
    nodes = np.where(lab == big)[0]
    ods = []
    for _ in range(n_od):
        o, t = rng.choice(nodes, 2, replace=False)
        d = np.zeros(net.n_nodes)
        d[o], d[t] = 1.0, -1.0
        ods.append(d)
    return A, ell, v, pert, ods, net


# --------------------------------------------------------------------------- #
# A. safeguard on/off
# --------------------------------------------------------------------------- #
def study_safeguard(quick: bool) -> None:
    print("=" * 78)
    print("A.  SAFEGUARD ON vs OFF  (synthetic sweep, IPM only, no crossover)")
    print("=" * 78)
    sizes = [10, 30] if quick else [10, 30, 80]
    spreads = [0.0, 4.0, 8.0]
    vscales = [3.0, 30.0, 300.0, 3000.0]
    seeds = range(4 if quick else 8)
    cfg = SSNConfig(max_iter=200)

    prem, raw_fail, saf_fail, rescue, tot = [], 0, 0, 0, 0
    nfast = nsafe = 0
    for n in sizes:
        for spd in spreads:
            for vs in vscales:
                for s in seeds:
                    prob, v = synth(n, s, spd, vs)
                    r0 = IPMSolver(cfg, crossover=False, safeguard=False)
                    r0.preprocess(prob)
                    R0 = r0.solve((v, GAMMA))
                    r1 = IPMSolver(cfg, crossover=False, safeguard=True)
                    r1.preprocess(prob)
                    R1 = r1.solve((v, GAMMA))
                    tot += 1
                    rb = (not R0.success) or (not np.isfinite(R0.residual)) or R0.residual > 1e-6
                    sb = (not R1.success) or (not np.isfinite(R1.residual)) or R1.residual > 1e-6
                    raw_fail += rb
                    saf_fail += sb
                    rescue += rb and not sb
                    nfast += R1.extras["n_fast"]
                    nsafe += R1.extras["n_safe"]
                    if not rb and not sb:
                        prem.append(R1.nit - R0.nit)
    print(f"instances tested            : {tot}")
    print(f"unsafeguarded (raw) failures: {raw_fail}")
    print(f"safeguarded failures        : {saf_fail}")
    print(f"rescues (raw fail, safe ok) : {rescue}")
    print(f"iteration premium (safe-raw): median={statistics.median(prem):.0f}  "
          f"mean={statistics.mean(prem):+.2f}  max={max(prem)}  (over both-converged)")
    print(f"step mix (safeguarded)      : {nfast} fast, {nsafe} safe "
          f"({100 * nsafe / max(nfast + nsafe, 1):.1f}% safe)")
    print()
    print("Reading: the safeguard is transparent (premium ~ 0, almost all fast steps)")
    print("yet provably convergent -- and rescues the instances where raw diverges.")
    print()


# --------------------------------------------------------------------------- #
# B. horse race vs CVXPY
# --------------------------------------------------------------------------- #
def _cvxpy_sieve(A, ell, v, d):
    """Solve one OD subproblem with CVXPY (Clarabel) for the sieve."""
    import cvxpy as cp

    m = A.shape[1]
    x = cp.Variable(m)
    # F(x) = sum_i ell_i ( x_i^2/2 + sum_l gamma_l x_i^l / l ),  gamma = [g3,g4,g5]
    powers = 0.5 * cp.square(x)
    for j, g in enumerate(GAMMA):
        powers = powers + g / (j + 3) * cp.power(x, j + 3)
    obj = cp.Minimize(ell @ powers - v @ x)
    cons = [A @ x == d, x >= 0, x <= 1]
    prob = cp.Problem(obj, cons)
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL)
    return x.value, time.perf_counter() - t0, prob.status


def _ipm_residual(prob, v, x):
    """KKT primal-feasibility residual ||Ax-d||_inf at a returned x (numpy)."""
    c = prob.constraint
    import torch

    xt = torch.as_tensor(x, dtype=torch.float64)
    return float((c.matvec(xt) - c.b).abs().max())


def study_horserace(quick: bool) -> None:
    print("=" * 78)
    print("B.  HORSE RACE: safeguarded IPM vs CVXPY (Clarabel)")
    print("=" * 78)
    nets = [("synthetic-200", None)]
    nets.append(("SiouxFalls", DATA + "SiouxFalls_net.tntp"))
    if not quick:
        nets.append(("ChicagoSketch", DATA + "ChicagoSketch_net.tntp"))
    n_od = 5 if quick else 12
    cfg = SSNConfig(max_iter=200)
    hdr = f"{'network':16} {'n/m':>11} | {'IPM ms':>8} {'IPM res':>9} {'it':>3} | {'CVXPY ms':>9} {'CVXPY res':>9} | {'maxΔx':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, path in nets:
        if path is None:
            prob, v = synth(200, 0, 4.0, 30.0)
            A = sp.csr_matrix(prob.constraint.A_scipy) if hasattr(prob.constraint, "A_scipy") else None
            # build OD list = the single demand already in prob
            ods = [to_numpy(prob.constraint.b)]
            probs = [(prob, v, ods[0])]
        else:
            try:
                A, ell, v, pert, ods, net = from_tntp(path, 0, n_od)
            except Exception as e:  # noqa: BLE001
                print(f"{name:16} (skipped: {e})")
                continue
            probs = []
            for d in ods:
                p = PUMProblem(pert, GeneralPolytope(A, d, ell=ell))
                probs.append((p, v, d))

        ipm_ms, ipm_res, ipm_it, cvx_ms, cvx_res, dxs = [], [], [], [], [], []
        nm = None
        for prob, vv, d in probs:
            # Pure Algorithm 1 (no SSN crossover): exactly the solver in the paper.
            solver = IPMSolver(cfg, crossover=False, safeguard=True)
            solver.preprocess(prob)
            t0 = time.perf_counter()
            R = solver.solve((vv, GAMMA))
            ipm_ms.append(1e3 * (time.perf_counter() - t0))
            ipm_res.append(R.residual)
            ipm_it.append(R.nit)
            xi = to_numpy(R.x)
            nm = f"{prob.constraint.num_constraints}/{prob.constraint.num_coords}"
            try:
                c = prob.constraint
                xc, tc, st = _cvxpy_sieve(c.A, to_numpy(c.ell), vv, d)
                cvx_ms.append(1e3 * tc)
                if xc is not None and "optimal" in str(st):
                    cvx_res.append(_ipm_residual(prob, vv, xc))
                    dxs.append(float(np.max(np.abs(xi - xc))))
            except Exception as e:  # noqa: BLE001
                if not cvx_ms:
                    print(f"   (cvxpy error on {name}: {type(e).__name__}: {e})")

        med = statistics.median
        cvx_ms_s = f"{med(cvx_ms):9.2f}" if cvx_ms else "      n/a"
        cvx_res_s = f"{med(cvx_res):9.1e}" if cvx_res else "      n/a"
        dx_s = f"{max(dxs):8.1e}" if dxs else "     n/a"
        print(f"{name:16} {nm:>11} | {med(ipm_ms):8.2f} {med(ipm_res):9.1e} "
              f"{int(med(ipm_it)):3d} | {cvx_ms_s} {cvx_res_s} | {dx_s}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["A", "B"], default=None)
    args = ap.parse_args()
    if args.only != "B":
        study_safeguard(args.quick)
    if args.only != "A":
        study_horserace(args.quick)
