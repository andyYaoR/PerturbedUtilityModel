"""
Comprehensive crosscheck of the unified AutoSolver across the whole TNTP corpus.

For every TNTP road network we solve the single-OD PURC forward problem with the
unified :class:`AutoSolver` (provably-ruled IPM / SSN regime) for each
perturbation (quadratic, Shannon entropy, modified entropy, polynomial sieve) and
utility scale ``beta``, and cross-check against CVXPY (Clarabel) wherever the
program is DCP-expressible and the network is small enough for CVXPY to be
practical.

We report, per (network, perturbation, beta):
  * AutoSolver convergence (how many of the OD-pairs converged), regime, median
    Newton iterations, median solve time, and median KKT residual;
  * the CVXPY time and the max primal discrepancy AutoSolver-vs-CVXPY.

OD-pairs are drawn from the largest *strongly*-connected component so a unit
o->d flow is always feasible (also on the asymmetric / disconnected networks).

Run:
    python dev/crosscheck_tntp.py                 # all networks, all perturbations
    python dev/crosscheck_tntp.py --cvxpy-max-m 8000
    python dev/crosscheck_tntp.py --only SiouxFalls,Anaheim,Barcelona
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import time

import numpy as np
import scipy.sparse.csgraph as csg

from purcsolver import PUMProblem, SSNConfig
from purcsolver.constraints import GeneralPolytope
from purcsolver.perturbations import get_perturbation
from purcsolver.solvers import get_solver
from purcsolver.utils.torch_compat import to_numpy
from tntp import load_net

import cvxpy as cp  # noqa: E402

PUM_NET = "/Users/ruiyao/Library/CloudStorage/Dropbox/Technion/Codes/PUM/examples/network"
EXTRA = {
    "Chicago-Regional": "/Users/ruiyao/Library/CloudStorage/Dropbox/Technion/Codes/"
    "LaplacianSolve/examples/data/ChicagoRegional_net.tntp",
}

# perturbation -> (constructor, cvxpy elementwise h or None, gamma)
PERTS = {
    "quad": (lambda: get_perturbation("quadratic"), lambda x: 0.5 * cp.square(x), np.zeros(0)),
    "entropy": (lambda: get_perturbation("entropy"), lambda x: -cp.entr(x), np.zeros(0)),
    "mod_entr": (
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


def _discover():
    """Return ``{name: path}`` for every TNTP _net file (PUM corpus + extras)."""
    nets = {}
    for f in sorted(glob.glob(os.path.join(PUM_NET, "*", "*_net.tntp"))) + sorted(
        glob.glob(os.path.join(PUM_NET, "*", "*_Net.tntp"))
    ):
        nets[os.path.basename(os.path.dirname(f))] = f
    nets.update(EXTRA)
    return nets


def _strong_component_nodes(A):
    """Node indices of the largest strongly-connected component of the digraph."""
    # directed adjacency: edge init -> term for each column of the incidence A
    m = A.shape[1]
    Acsc = A.tocsc()
    init = np.asarray((Acsc == 1).argmax(axis=0)).ravel()
    term = np.asarray((Acsc == -1).argmax(axis=0)).ravel()
    adj = csg.csgraph_from_dense(np.zeros((1, 1)))  # placeholder
    import scipy.sparse as sp

    adj = sp.csr_matrix((np.ones(m), (init, term)), shape=(A.shape[0], A.shape[0]))
    ncomp, labels = csg.connected_components(adj, directed=True, connection="strong")
    largest = np.bincount(labels).argmax()
    return np.flatnonzero(labels == largest)


def _ods(A, n_od, seed=0):
    """Sample feasible OD-pairs from the largest strongly-connected component."""
    pool = _strong_component_nodes(A)
    rng = np.random.default_rng(seed)
    ods = []
    if len(pool) < 2:
        return ods
    while len(ods) < n_od:
        o, d = int(rng.choice(pool)), int(rng.choice(pool))
        if o != d:
            ods.append((o, d))
    return ods


def _cost(net):
    """
    Model-valid link weight ``ell`` / cost base.

    Centroid-connector links carry zero recorded length (and zero free-flow time);
    the perturbed-utility model needs a strictly positive, *non-degenerate* weight
    ``ell_i > 0`` (else those coordinates are unperturbed -> the objective is
    linear there and the dual Newton Laplacian is catastrophically
    ill-conditioned).  So assign zero-length links the median link length rather
    than a near-zero floor, and floor tiny positives at median/1000 to keep the
    weight spread (hence the Laplacian condition number) bounded.
    """
    length = np.asarray(net.length, dtype=float)
    pos = length[length > 0]
    if pos.size == 0:
        ff = np.asarray(net.fftt, dtype=float)
        pos = ff[ff > 0]
    fill = float(np.median(pos)) if pos.size else 1.0
    cost = np.where(length > 0, length, fill)
    return np.maximum(cost, fill * 1e-3)


def run(names, n_od, cvxpy_max_m, betas):
    """Run the crosscheck and print a per-network table + a final summary."""
    nets = _discover()
    if names:
        nets = {k: v for k, v in nets.items() if k in names}
    order = sorted(nets, key=lambda k: load_net(nets[k]).n_links)

    total, converged, beaten, cvxpy_cmp = 0, 0, 0, 0
    failures = []
    print(f"{'network':28s} {'m':>6s} {'pert':8s} {'b':>2s} {'reg':>4s} "
          f"{'conv':>4s} {'nit':>4s} {'auto ms':>8s} {'resid':>7s} {'cvxpy ms':>9s} {'spd':>6s} {'vs cvx':>7s}")
    for name in order:
        net = load_net(nets[name])
        A, n, m = net.A, net.n_nodes, net.n_links
        cost = _cost(net)
        ods = _ods(A, n_od, seed=1)
        if not ods:
            print(f"{name:28s} {m:6d}  (no feasible OD-pairs)")
            continue
        for pname, (make_pert, cvxpy_h, gamma) in PERTS.items():
            for beta in betas:
                v = -beta * cost
                prob = PUMProblem(make_pert(), GeneralPolytope(A, np.zeros(n), ell=cost, validate=False))
                solver = get_solver("auto", config=SSNConfig(tol=1e-9, max_iter=400))
                solver.preprocess(prob)

                # The first OD per cell is an untimed warmup (absorbs one-time
                # torch / CHOLMOD / symbolic-factorization init and the lazy SDDM
                # build) so the reported time is the steady-state per-solve cost
                # an estimation loop would see.  Convergence is still counted on it.
                ms, nits, resids, conv = [], [], [], 0
                regime = solver.regime
                xsol = {}
                for i, (o, d) in enumerate(ods):
                    bvec = np.zeros(n)
                    bvec[o], bvec[d] = 1.0, -1.0
                    t = time.perf_counter()
                    res = solver.solve((v, gamma), b=bvec, lam0=np.zeros(n))
                    dt = (time.perf_counter() - t) * 1e3
                    regime = res.extras.get("regime", regime)
                    if not res.success:
                        continue
                    conv += 1
                    nits.append(res.nit)
                    resids.append(res.residual)
                    xsol[(o, d)] = to_numpy(res.x).ravel()
                    if i > 0:
                        ms.append(dt)

                total += 1
                if conv == len(ods):
                    converged += 1
                else:
                    failures.append(f"{name}/{pname}/b{beta:g} ({conv}/{len(ods)})")

                # CVXPY crosscheck where expressible + tractable (first OD untimed too).
                cvx_ms, cvx_err = None, None
                if cvxpy_h is not None and m <= cvxpy_max_m:
                    cvt, cerr = [], []
                    for i, (o, d) in enumerate(ods):
                        bvec = np.zeros(n)
                        bvec[o], bvec[d] = 1.0, -1.0
                        x = cp.Variable(m)
                        pr = cp.Problem(
                            cp.Minimize(cp.sum(cp.multiply(cost, cvxpy_h(x))) - v @ x),
                            [A @ x == bvec, x >= 0, x <= 1],
                        )
                        try:
                            tt = time.perf_counter()
                            pr.solve(solver=cp.CLARABEL)
                            el = (time.perf_counter() - tt) * 1e3
                            if i > 0:
                                cvt.append(el)
                            if x.value is not None and (o, d) in xsol:
                                cerr.append(float(np.abs(xsol[(o, d)] - np.asarray(x.value)).max()))
                        except cp.error.SolverError:
                            pass
                    if cvt:
                        cvx_ms = statistics.median(cvt)
                        cvx_err = statistics.median(cerr) if cerr else None

                med_ms = statistics.median(ms) if ms else float("nan")
                spd = (cvx_ms / med_ms) if (cvx_ms and med_ms == med_ms) else None
                if spd is not None:
                    cvxpy_cmp += 1
                    if spd >= 1.0:
                        beaten += 1
                print(
                    f"{name:28s} {m:6d} {pname:8s} {beta:2.0f} {regime:>4s} "
                    f"{conv:d}/{len(ods):<2d} {int(statistics.median(nits)) if nits else 0:>4d} "
                    f"{med_ms:8.1f} {statistics.median(resids) if resids else float('nan'):7.0e} "
                    f"{('%.1f' % cvx_ms) if cvx_ms else '-':>9s} {('%.1fx' % spd) if spd else '-':>6s} "
                    f"{('%.0e' % cvx_err) if cvx_err is not None else '-':>7s}",
                    flush=True,
                )

    print(f"\n=== SUMMARY ===")
    print(f"cells: {total}   AutoSolver fully-converged: {converged}/{total}")
    print(f"CVXPY-comparable cells: {cvxpy_cmp}   AutoSolver >= CVXPY speed: {beaten}/{cvxpy_cmp}")
    if failures:
        print(f"NON-converged cells ({len(failures)}): " + ", ".join(failures))
    else:
        print("NO non-converged cells: AutoSolver solved every network x perturbation x scale.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-od", type=int, default=3)
    ap.add_argument("--cvxpy-max-m", type=int, default=12000)
    ap.add_argument("--betas", default="1,5")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    run(
        [s for s in args.only.split(",") if s],
        args.n_od,
        args.cvxpy_max_m,
        [float(b) for b in args.betas.split(",")],
    )
