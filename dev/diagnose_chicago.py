"""
Per-OD diagnostics for Chicago PURC convergence failures.

This script is intentionally separate from ``bench_chicago.py``: benchmarks
summarize timings, while this reports the iteration facts needed to debug the
globalization phase.

Run:
    python dev/diagnose_chicago.py regional --perturbation sieve --beta 5
    python dev/diagnose_chicago.py sketch --perturbation quadratic --beta 5
"""

from __future__ import annotations

import argparse

import numpy as np

from purcsolver import PUMProblem, SSNConfig
from purcsolver.constraints import GeneralPolytope
from purcsolver.perturbations import get_perturbation
from purcsolver.solvers import RegularizedSSNSolver
from tntp import load_net

DATA = "/Users/ruiyao/Library/CloudStorage/Dropbox/Technion/Codes/LaplacianSolve/examples/data/"
NETS = {
    "sketch": DATA + "ChicagoSketch_net.tntp",
    "regional": DATA + "ChicagoRegional_net.tntp",
}


def feasible_ods(n_nodes: int, n_od: int, seed: int = 1) -> list[tuple[int, int]]:
    """Return deterministic OD pairs matching the benchmark generator."""
    rng = np.random.default_rng(seed)
    ods = []
    while len(ods) < n_od:
        o, d = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
        if o != d:
            ods.append((o, d))
    return ods


def perturbation_spec(name: str):
    """Return ``(display_name, perturbation, gamma)`` for a diagnostic kernel."""
    if name == "quadratic":
        return "quadratic", get_perturbation("quadratic"), np.zeros(0)
    if name == "entropy":
        return "entropy", get_perturbation("entropy"), np.zeros(0)
    if name == "sieve":
        gamma = np.array([0.3, 0.1])
        return "sieve(L=4)", get_perturbation("polynomial_sieve", gamma=gamma), gamma
    raise ValueError(f"unknown perturbation {name!r}")


def summarize(values: list[float], n: int = 8) -> str:
    """Format the first/last entries of a diagnostic history."""
    if not values:
        return "[]"
    if len(values) <= 2 * n:
        return np.array2string(np.asarray(values), precision=3, suppress_small=False)
    first = np.array2string(np.asarray(values[:n]), precision=3, suppress_small=False)
    last = np.array2string(np.asarray(values[-n:]), precision=3, suppress_small=False)
    return f"{first} ... {last}"


def run(args: argparse.Namespace) -> None:
    """Run diagnostics for one network/kernel/utility scale."""
    net = load_net(NETS[args.network])
    cost = np.maximum(net.length, 1e-3)
    title, perturbation, gamma = perturbation_spec(args.perturbation)
    v = -args.beta * cost
    ods = feasible_ods(net.n_nodes, args.n_od, seed=args.seed)
    poly = GeneralPolytope(net.A, np.zeros(net.n_nodes), ell=cost, validate=False)
    prob = PUMProblem(perturbation, poly)

    print(
        f"Chicago {args.network}: n={net.n_nodes}, m={net.n_links}, "
        f"perturbation={title}, beta={args.beta:g}, max_iter={args.max_iter}"
    )
    print(f"OD pairs: {ods}")

    for idx, (origin, destination) in enumerate(ods):
        b = np.zeros(net.n_nodes)
        b[origin] = 1.0
        b[destination] = -1.0
        solver = RegularizedSSNSolver(SSNConfig(tol=args.tol, max_iter=args.max_iter))
        solver.preprocess(prob)
        res = solver.solve((v, gamma), b=b, lam0=np.zeros(net.n_nodes))
        hist = res.residual_history
        eps = res.extras.get("lm_eps_trace", [])
        best_resid = min(hist) if hist else res.residual
        print(
            f"\nOD {idx}: {origin}->{destination} success={res.success} "
            f"status={res.status} nit={res.nit} residual={res.residual:.3e} "
            f"best={best_resid:.3e} active={res.extras.get('active_set_size')} "
            f"backend={res.extras.get('backend_method')} phase={res.extras.get('backend_phase')}"
        )
        print(f"  residual history: {summarize(hist)}")
        print(f"  lm eps trace:     {summarize(eps)}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", choices=sorted(NETS), nargs="?", default="regional")
    parser.add_argument(
        "--perturbation",
        choices=["quadratic", "entropy", "sieve"],
        default="sieve",
    )
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--n-od", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
