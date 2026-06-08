"""
Solve a perturbed-utility route-choice problem on the SiouxFalls network.

This example demonstrates the high-level forward-solver interface. Given a road
network and a perturbation kernel, the perturbed-utility model predicts the
optimal response:

    x*(theta) = argmin_{A x = b, 0 <= x <= 1}  F(x; gamma) - v(beta)^T x,

where ``A`` is the node-arc incidence matrix, ``F(x) = sum_i ell_i h(x_i; gamma)``
is a separable, strictly convex perturbation (here weighted by the free-flow travel
time ``ell_i``), and ``v(beta)`` are the link utilities.

The example loads the SiouxFalls network (24 nodes, 76 directed links), builds a
:class:`~purc.static_purc.PUMProblem`, and solves it through the unified interface
:func:`~purc.static_purc.get_solver` entry point. The ``"auto"`` solver chooses its
numerical regime from a provable property of the kernel. It then solves a single 
origin-destination demand and a batch of demands, reusing a single preprocessing step.

Run:  python examples/forward_solve.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

from purc.static_purc import ForwardSolverConfig, PUMProblem, get_perturbation, get_solver
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.utils.torch_compat import to_numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tntp import load_net  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "SiouxFalls_net.tntp")


def _demand(n: int, origin: int, dest: int) -> np.ndarray:
    """Return a unit OD demand vector (``+1`` at origin, ``-1`` at destination)."""
    b = np.zeros(n)
    b[origin], b[dest] = 1.0, -1.0
    return b


def main() -> None:
    """Build the SiouxFalls forward problem and solve it (single + batched)."""
    net = load_net(_DATA)
    A, n, m = net.A, net.n_nodes, net.n_links
    ell = net.fftt  # per-link free-flow time: the separable weight ell_i > 0
    print(f"SiouxFalls: {n} nodes, {m} directed links; ell = free-flow time.")

    # Utility v = -ell (Z=None -> v = beta directly): shorter links are preferred.
    beta = -ell
    gamma = np.zeros(0)  # modified entropy is parameter-free

    # One reference demand fixes the polytope; solve()/solve_batch() override it.
    origin, dest = 0, 12
    b0 = _demand(n, origin, dest)
    cons = GeneralPolytope(A, b0, lo=0.0, hi=1.0, ell=ell)
    prob = PUMProblem(get_perturbation("modified_entropy"), cons, Z=None)

    solver = get_solver("auto", config=ForwardSolverConfig(tol=1e-9, max_iter=200))
    solver.preprocess(prob)  # one-time setup
    print(f"Solver regime (selected from a provable kernel property): {solver.regime!r}\n")

    # --- single OD solve -----------------------------------------------------
    res = solver.solve((beta, gamma), b=b0)
    x = to_numpy(res.x)
    active = int((x > 1e-8).sum())
    fstar = float(res.conjugate) if res.conjugate is not None else float("nan")
    print(f"Single solve (origin {origin} -> destination {dest}):")
    print(f"  success={res.success}  status={res.status} ({res.message})")
    print(f"  newton_iters={res.nit}  residual ||Ax-b||inf={res.residual:.2e}")
    print(f"  F*(v;gamma)={fstar:.6f}  active links={active}/{m}  "
          f"flow mass sum(x)={float(x.sum()):.4f}\n")

    # --- batched OD solve (one preprocess, B systems at once) ----------------
    rng = np.random.default_rng(0)
    B = 8
    b_batch = np.zeros((B, n))
    for i in range(B):
        o, d = rng.choice(n, 2, replace=False)
        b_batch[i, int(o)], b_batch[i, int(d)] = 1.0, -1.0
    resb = solver.solve_batch((beta, gamma), b_batch)
    xb = to_numpy(resb.x)  # [B, N]
    max_resid = max(float(np.abs(A @ xb[i] - b_batch[i]).max()) for i in range(B))
    print(f"Batched solve ({B} demands, vectorized over systems):")
    print(f"  all converged={resb.success}  max newton_iters={resb.nit}  "
          f"max residual={max_resid:.2e}")
    print("\nForward solve completed on SiouxFalls (modified-entropy kernel).")


if __name__ == "__main__":
    main()
