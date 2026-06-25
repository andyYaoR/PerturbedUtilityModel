"""
Benchmark: PURCSolver forward solve vs. CVXPY (Clarabel) on bundled road networks.

For each TNTP network in ``examples/data`` this times the perturbed-utility forward
solve -- the interior-point method (IPM) -- against the *same* convex program solved
by CVXPY's Clarabel, on a DCP-expressible kernel (modified entropy), and reports the
median per-solve wall time, the speedup, and the maximum primal discrepancy (a
correctness check; the two should agree to solver tolerance).

The solver is built once with ``preprocess`` and reused across origin-destination
pairs (warm starts + a cached symbolic factorization) -- the way an assignment or
estimation loop uses it -- so the first solve per network is an untimed warm-up.

CVXPY is optional (``pip install -e ".[oracle]"``); without it the PURC timings are
still reported.

Run:
    python benchmarks/forward_vs_cvxpy.py
    python benchmarks/forward_vs_cvxpy.py --networks SiouxFalls,ChicagoSketch --n-od 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from purc.static_purc import ForwardSolverConfig, PUMProblem, get_perturbation, get_solver
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.utils.torch_compat import to_numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from tntp import load_net  # noqa: E402

try:
    import cvxpy as cp

    _HAS_CVXPY = True
except ImportError:  # pragma: no cover - the oracle extra is optional
    _HAS_CVXPY = False

_DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data")


def _cost(net) -> np.ndarray:
    """Strictly-positive per-link weight (length, with zero-length links filled)."""
    length = np.asarray(net.length, dtype=float)
    pos = length[length > 0]
    fill = float(np.median(pos)) if pos.size else 1.0
    cost = np.where(length > 0, length, fill)
    return np.maximum(cost, fill * 1e-3)


def _strong_component_nodes(a: sp.spmatrix) -> np.ndarray:
    """Node indices of the largest strongly-connected component of the digraph."""
    acsc = a.tocsc()
    init = np.asarray((acsc == 1).argmax(axis=0)).ravel()
    term = np.asarray((acsc == -1).argmax(axis=0)).ravel()
    adj = sp.csr_matrix((np.ones(a.shape[1]), (init, term)), shape=(a.shape[0], a.shape[0]))
    _, labels = csg.connected_components(adj, directed=True, connection="strong")
    return np.flatnonzero(labels == np.bincount(labels).argmax())


def _ods(a: sp.spmatrix, n_od: int, seed: int = 0) -> list[tuple[int, int]]:
    """Sample feasible OD pairs from the largest strongly-connected component."""
    pool = _strong_component_nodes(a)
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int]] = []
    while len(out) < n_od and len(pool) >= 2:
        o, d = int(rng.choice(pool)), int(rng.choice(pool))
        if o != d:
            out.append((o, d))
    return out


def _demand(n: int, o: int, d: int) -> np.ndarray:
    """Unit OD demand (+1 at origin, -1 at destination)."""
    b = np.zeros(n)
    b[o], b[d] = 1.0, -1.0
    return b


def _bench_purc(a, cost, v, ods):
    """Solve each OD with the IPM (warm-started); return (median_ms, {od: x})."""
    n = a.shape[0]
    prob = PUMProblem(get_perturbation("modified_entropy"), GeneralPolytope(a, _demand(n, *ods[0]), ell=cost))
    solver = get_solver("ipm", config=ForwardSolverConfig(tol=1e-9, max_iter=200))
    solver.preprocess(prob)
    times, sols = [], {}
    for i, (o, d) in enumerate(ods):
        b = _demand(n, o, d)
        t = time.perf_counter()
        res = solver.solve((v, np.zeros(0)), b=b)
        dt = (time.perf_counter() - t) * 1e3
        sols[(o, d)] = to_numpy(res.x).ravel()
        if i > 0:  # first solve is an untimed warm-up
            times.append(dt)
    return (statistics.median(times) if times else float("nan")), sols


def _bench_batched(a, cost, v, ods, warmup=8):
    """
    Solve all ODs in one GIL-released parallel ``solve_batch`` call.

    Returns ``(seconds, x)`` with ``x`` the ``[B, N]`` stacked primal solution.
    """
    n = a.shape[0]
    b_batch = np.stack([_demand(n, o, d) for (o, d) in ods])
    prob = PUMProblem(get_perturbation("modified_entropy"), GeneralPolytope(a, b_batch[0], ell=cost))
    solver = get_solver("ipm", config=ForwardSolverConfig(tol=1e-9, max_iter=200))
    solver.preprocess(prob)
    if warmup:
        solver.solve_batch((v, np.zeros(0)), b_batch[:warmup])  # untimed warm-up
    t = time.perf_counter()
    res = solver.solve_batch((v, np.zeros(0)), b_batch)  # one parallel solve over all ODs
    return (time.perf_counter() - t), to_numpy(res.x)


def _bench_cvxpy(a, cost, v, ods):
    """Solve each OD with CVXPY/Clarabel; return (median_ms, {od: x})."""
    n, m = a.shape
    times, sols = [], {}
    for i, (o, d) in enumerate(ods):
        b = _demand(n, o, d)
        x = cp.Variable(m)
        prob = cp.Problem(
            cp.Minimize(cp.sum(cp.multiply(cost, -cp.entr(1.0 + x) - x)) - v @ x),
            [a @ x == b, x >= 0, x <= 1],
        )
        t = time.perf_counter()
        prob.solve(solver=cp.CLARABEL)
        dt = (time.perf_counter() - t) * 1e3
        if x.value is not None:
            sols[(o, d)] = np.asarray(x.value).ravel()
        if i > 0:
            times.append(dt)
    return (statistics.median(times) if times else float("nan")), sols


def main() -> None:
    """Run the benchmark over the requested networks and print a table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", default="SiouxFalls,ChicagoSketch")
    ap.add_argument("--n-od", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-od", type=int, default=0,
                    help="if >0, also time one parallel solve_batch over this many ODs")
    args = ap.parse_args()

    print(f"Forward solve (modified entropy): IPM vs CVXPY/Clarabel | n_od={args.n_od}")
    if not _HAS_CVXPY:
        print("(CVXPY not installed -- reporting IPM timings only; `pip install -e \".[oracle]\"`)")
    print(f"\n{'network':16s} {'nodes':>6s} {'links':>6s} {'IPM ms':>9s} "
          f"{'CVXPY ms':>9s} {'speedup':>8s} {'max|Δx|':>9s}")
    for name in (s for s in args.networks.split(",") if s):
        net = load_net(os.path.join(_DATA, f"{name}_net.tntp"))
        a, cost = net.A, _cost(net)
        ods = _ods(a, args.n_od + 1, seed=args.seed)  # +1 for the warm-up
        if len(ods) < 2:
            print(f"{name:16s}  (no feasible OD pairs)")
            continue
        v = -cost  # utility prefers short links
        purc_ms, purc_x = _bench_purc(a, cost, v, ods)
        if _HAS_CVXPY:
            cvx_ms, cvx_x = _bench_cvxpy(a, cost, v, ods)
            shared = [k for k in purc_x if k in cvx_x]
            dmax = max((np.abs(purc_x[k] - cvx_x[k]).max() for k in shared), default=float("nan"))
            spd = f"{cvx_ms / purc_ms:.1f}x" if purc_ms == purc_ms and purc_ms > 0 else "-"
            print(f"{name:16s} {net.n_nodes:>6d} {net.n_links:>6d} {purc_ms:>9.2f} "
                  f"{cvx_ms:>9.1f} {spd:>8s} {dmax:>9.1e}")
        else:
            print(f"{name:16s} {net.n_nodes:>6d} {net.n_links:>6d} {purc_ms:>9.2f} "
                  f"{'-':>9s} {'-':>8s} {'-':>9s}")

    if args.batch_od <= 0:
        return
    print(f"\nBatched throughput: one parallel solve_batch over {args.batch_od} OD pairs "
          "(CVXPY≈ extrapolates its per-solve cost x ODs -- it cannot batch)")
    print(f"\n{'network':16s} {'ODs':>6s} {'total s':>8s} {'per-OD ms':>10s} "
          f"{'OD/s':>7s} {'CVXPY≈':>9s} {'speedup':>8s} {'max|Δx|':>9s}")
    for name in (s for s in args.networks.split(",") if s):
        net = load_net(os.path.join(_DATA, f"{name}_net.tntp"))
        a, cost = net.A, _cost(net)
        ods = _ods(a, args.batch_od, seed=args.seed)
        if len(ods) < 2:
            print(f"{name:16s}  (no feasible OD pairs)")
            continue
        v = -cost
        secs, xb = _bench_batched(a, cost, v, ods)
        per_od, ods_per_s = secs / len(ods) * 1e3, len(ods) / secs
        cvx_str, spd_str, dmax_str = "-", "-", "-"
        if _HAS_CVXPY:
            cvx_ms, cvx_x = _bench_cvxpy(a, cost, v, ods[: min(5, len(ods))])
            cvx_total = cvx_ms * 1e-3 * len(ods)  # CVXPY cannot batch: B independent solves
            idx = {od: i for i, od in enumerate(ods)}
            dmax = max((np.abs(xb[idx[od]] - cvx_x[od]).max() for od in cvx_x), default=float("nan"))
            cvx_str, spd_str, dmax_str = f"{cvx_total:.0f}s", f"{cvx_total / secs:.1f}x", f"{dmax:.1e}"
        print(f"{name:16s} {len(ods):>6d} {secs:>8.2f} {per_od:>10.3f} "
              f"{ods_per_s:>7.0f} {cvx_str:>9s} {spd_str:>8s} {dmax_str:>9s}")


if __name__ == "__main__":
    main()
