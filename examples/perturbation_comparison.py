"""
Compare how different perturbation kernels shape route choice.

The perturbed-utility model distributes demand across routes by adding a strictly
convex perturbation ``F(x) = sum_i ell_i h(x_i; gamma)`` to the negative utility.
The shape of the kernel ``h`` determines how sharply travelers concentrate on the
cheaper route:

  - quadratic:        ``h(x) = x^2 / 2``;
  - Shannon entropy:  ``h(x) = x log x``; 
  - modified entropy: ``h(x) = (1 + x) log(1 + x) - x``;
  - polynomial sieve: ``h(x) = x^2 / 2 + gamma_3 x^3 / 3`.

The example solves the same origin-destination problem under each kernel on a small
network with two-node routes, one slightly cheaper than the other, and
reports how the unit demand is split between them.

Run:  python examples/perturbation_comparison.py
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from purc.static_purc import ForwardSolverConfig, PUMProblem, get_perturbation, get_solver
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.utils.torch_compat import to_numpy

# Two routes through distinct intermediate nodes (1 and 2):
#   route A:  0 -(arc0)-> 1 -(arc1)-> 3
#   route B:  0 -(arc2)-> 2 -(arc3)-> 3
# Each arc connects a distinct node pair -> no parallel links.
_ARCS = [(0, 1), (1, 3), (0, 2), (2, 3)]
# Per-link travel cost: route A (arcs 0,1) is cheaper than route B (arcs 2,3).
_COST = np.array([1.0, 1.0, 1.3, 1.3])

# (registry name, extra kwargs) for the kernels to compare.
_KERNELS = [
    ("quadratic", {}),
    ("shannon_entropy", {}),
    ("modified_entropy", {}),
    ("polynomial_sieve", {"gamma": np.array([0.5])}),
]


def _incidence() -> sp.csr_matrix:
    """Build the 4-node, 4-arc node-arc incidence (``+1`` tail, ``-1`` head)."""
    A = np.zeros((4, len(_ARCS)))
    for k, (tail, head) in enumerate(_ARCS):
        A[tail, k], A[head, k] = 1.0, -1.0
    return sp.csr_matrix(A)


def _solve_split(name: str, kwargs: dict, A: sp.csr_matrix, b: np.ndarray):
    """Solve the OD problem under one kernel; return (regime, flow_A, flow_B, x)."""
    gamma = np.asarray(kwargs.get("gamma", np.zeros(0)), dtype=float)
    cons = GeneralPolytope(A, b, lo=0.0, hi=1.0, ell=np.ones(len(_ARCS)))
    prob = PUMProblem(get_perturbation(name, **kwargs), cons, Z=None)
    solver = get_solver("auto", config=ForwardSolverConfig(tol=1e-9, max_iter=200))
    solver.preprocess(prob)
    res = solver.solve((-_COST, gamma), b=b)  # utility v = -cost
    x = to_numpy(res.x)
    # Flow conservation makes both arcs of a route carry the same flow:
    # route A flow = x[arc0], route B flow = x[arc2].
    return solver.regime, float(x[0]), float(x[2]), x, res.success


def main() -> None:
    """Compare the route split across perturbation kernels on the 2-route network."""
    A = _incidence()
    b = np.array([1.0, 0.0, 0.0, -1.0])  # one unit of demand, node 0 -> node 3
    print("Two node-disjoint routes 0->3; route A cost 2.0, route B cost 2.6.")
    print("Unit demand split by kernel (higher 'spread' = more even split):\n")
    print(f"{'kernel':18s} {'regime':>6s} {'route A':>8s} {'route B':>8s} "
          f"{'spread':>7s} {'active':>7s}")
    for name, kwargs in _KERNELS:
        regime, fa, fb, x, ok = _solve_split(name, kwargs, A, b)
        # Spread = entropy of the route-choice distribution, in bits (1.0 = 50/50).
        ps = np.array([fa, fb])
        ps = ps[ps > 0]
        spread = float(-(ps * np.log2(ps)).sum())
        active = int((x > 1e-8).sum())
        flag = "" if ok else "  (NOT CONVERGED)"
        print(f"{name:18s} {regime:>6s} {fa:>8.3f} {fb:>8.3f} "
              f"{spread:>7.3f} {active:>5d}/{len(_ARCS)}{flag}")
    print("\nShannon entropy keeps both routes busiest; modified entropy "
          "concentrates most on the cheaper route.")


if __name__ == "__main__":
    main()
