"""
Throughput of the batched IPM vs a per-OD loop and the batched SSN.

All three solve the *same* set of OD-pairs on a shared network at one ``theta``:

  * **IPM batch**   -- ``IPMSolver.solve_batch`` (vectorized Algorithm 1).
  * **IPM per-OD**  -- ``IPMSolver.solve`` called once per demand (the baseline).
  * **SSN batch**   -- ``RegularizedSSNSolver.solve_batch`` (the existing batch).

We report wall-clock per OD-pair and the speedup of the batch over the loop, plus
a parity check that the batched and per-OD primals agree.  Networks: a synthetic
graph and the TNTP road networks (Sioux Falls, Chicago Sketch).

Run:
    python dev/bench_ipm_batch.py            # default sizes
    python dev/bench_ipm_batch.py --quick    # fewer OD-pairs / smaller nets
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from purc.static_purc import PUMProblem, SSNConfig
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.solvers.ssn import RegularizedSSNSolver
from purc.static_purc.utils.torch_compat import to_numpy
from bench_safeguard import GAMMA, DATA, synth, from_tntp


def _time(fn, repeats: int = 3) -> float:
    """Median wall-clock (seconds) of ``fn`` over ``repeats`` runs (after a warmup)."""
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _build(name: str, path, n_od: int):
    """Return (incidence/general A, ell, v, list-of-demands) for a network."""
    if path is None:
        prob, v = synth(120, 0, 4.0, 30.0)
        c = prob.constraint
        # Make several OD pairs on the synthetic graph.
        n = c.num_constraints
        rng = np.random.default_rng(0)
        ods = []
        for _ in range(n_od):
            o, t = rng.choice(n, 2, replace=False)
            d = np.zeros(n)
            d[o], d[t] = 1.0, -1.0
            ods.append(d)
        return c.A, to_numpy(c.ell), v, ods, prob.perturbation
    A, ell, v, pert, ods, _net = from_tntp(path, 0, n_od)
    return A, ell, v, ods, pert


def main(quick: bool) -> None:
    """Run the batched-vs-loop throughput study."""
    nets = [("synthetic-120", None), ("SiouxFalls", DATA + "SiouxFalls_net.tntp")]
    if not quick:
        nets.append(("ChicagoSketch", DATA + "ChicagoSketch_net.tntp"))
    n_od = 8 if quick else 32
    cfg = SSNConfig(max_iter=200)
    hdr = (
        f"{'network':15} {'n/m':>11} {'B':>4} | {'IPM batch ms':>13} {'IPM loop ms':>12} "
        f"{'speedup':>8} | {'SSN batch ms':>13} | {'parity':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, path in nets:
        A, ell, v, ods, pert = _build(name, path, n_od)
        b_batch = np.stack(ods)
        prob = PUMProblem(pert, GeneralPolytope(A, ods[0], ell=ell))
        nm = f"{prob.constraint.num_constraints}/{prob.constraint.num_coords}"

        ipm = IPMSolver(cfg, crossover=False, safeguard=True)
        ipm.preprocess(prob)
        t_batch = _time(lambda: ipm.solve_batch((v, GAMMA), b_batch))
        xb = to_numpy(ipm.solve_batch((v, GAMMA), b_batch).x)

        def _loop():
            for d in ods:
                ipm.solve((v, GAMMA), b=d)

        t_loop = _time(_loop)
        # parity batch vs loop
        xl = np.stack([to_numpy(ipm.solve((v, GAMMA), b=d).x) for d in ods])
        parity = float(np.max(np.abs(xb - xl)))

        ssn = RegularizedSSNSolver(cfg)
        ssn.preprocess(prob)
        t_ssn = _time(lambda: ssn.solve_batch((v, GAMMA), b_batch))

        B = len(ods)
        speed = t_loop / t_batch if t_batch > 0 else float("nan")
        print(
            f"{name:15} {nm:>11} {B:>4} | {1e3 * t_batch:13.2f} {1e3 * t_loop:12.2f} "
            f"{speed:7.1f}x | {1e3 * t_ssn:13.2f} | {parity:9.1e}"
        )
    print()
    print("IPM batch vs per-OD loop: same algorithm, one vectorized native solve per step.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(args.quick)
