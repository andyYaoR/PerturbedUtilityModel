"""
SiouxFalls network: the PURC Laplacian solve step under a quadratic perturbation.

This example focuses *only* on the linear-solve step that the PURC
semismooth-Newton iteration performs at each iteration -- it does NOT run the
full Newton/equilibrium loop.

Background.  The PURC Newton system is

    (H^k + eps_k I) d_lambda = -r^k,   H^k = C_{S^k} diag(D^k_i) C_{S^k}^T,

where ``C`` is the node-link incidence, ``S^k`` the active links, and
``D^k_i = 1 / (l_i * h''(x_i; gamma))`` with ``l_i`` the link length and ``h``
the perturbation kernel.  For a **quadratic perturbation** ``h(xi) = 1/2 xi^2``
we have ``h'' = 1``, so the edge weights are *constant*, ``D_i = 1 / l_i``, and

    H = C diag(1/l_i) C^T

is a fixed weighted graph Laplacian of the network.  Adding the regularizer
``eps I`` makes ``M = H + eps I`` an SDDM (symmetric positive-definite) matrix,
which we solve with :class:`purc.laplaciansolve.SDDMSolver`.

We load the SiouxFalls network (24 nodes, 76 links), build ``H`` from the link
lengths, and solve ``M x = b`` for several right-hand sides -- reusing the one
cached factorization across all of them, exactly as the Newton loop would.

Run:  python examples/siouxfalls_laplacian_solve.py
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np
from scipy import sparse

from purc.laplaciansolve import SDDMSolver, SolverConfig
from purc.laplaciansolve.reference import lap

_DATA = os.path.join(os.path.dirname(__file__), "data", "SiouxFalls_net.tntp")


def load_tntp_links(path: str) -> Tuple[int, List[Tuple[int, int, float]]]:
    """Parse a TNTP ``_net`` file into ``(n_nodes, [(init, term, length), ...])``.

    Args:
        path: Path to the ``*_net.tntp`` file.

    Returns:
        The node count and the directed link list (1-based nodes, link length).
    """
    n_nodes = 0
    links: List[Tuple[int, int, float]] = []
    in_data = False
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.upper().startswith("<NUMBER OF NODES>"):
                n_nodes = int(line.split()[-1])
            if line.startswith("<END OF METADATA>"):
                in_data = True
                continue
            if not in_data or not line or line.startswith("~"):
                continue
            parts = line.replace(";", "").split()
            if len(parts) < 5:
                continue
            init, term = int(parts[0]), int(parts[1])
            length = float(parts[3])  # TNTP column 4 = length
            links.append((init, term, length))
    return n_nodes, links


def build_weighted_laplacian_adjacency(
    n_nodes: int, links: List[Tuple[int, int, float]]
) -> sparse.csc_matrix:
    """Build the symmetric adjacency with quadratic-perturbation weights ``1/l``.

    Directed links between the same pair are merged into one undirected edge
    (SiouxFalls stores both directions with equal length).

    Args:
        n_nodes: Number of network nodes.
        links: Directed links ``(init, term, length)`` (1-based nodes).

    Returns:
        The symmetric, zero-diagonal weighted adjacency (CSC).
    """
    weight = {}
    for init, term, length in links:
        u, v = init - 1, term - 1  # 0-based
        key = (min(u, v), max(u, v))
        weight[key] = 1.0 / length  # D_i = 1 / (l_i * h''),  h'' = 1 (quadratic)
    rows, cols, data = [], [], []
    for (u, v), w in weight.items():
        rows += [u, v]
        cols += [v, u]
        data += [w, w]
    return sparse.csc_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def main() -> None:
    """Build the SiouxFalls SDDM system and solve it with the reusable handle."""
    n_nodes, links = load_tntp_links(_DATA)
    adjacency = build_weighted_laplacian_adjacency(n_nodes, links)
    n_edges = adjacency.nnz // 2
    print(f"SiouxFalls: {n_nodes} nodes, {n_edges} undirected links "
          f"({len(links)} directed).")

    eps = 1e-6  # PURC regularizer eps_k (floor)
    h_lap = lap(adjacency)
    m = sparse.csc_matrix(h_lap + eps * sparse.eye(n_nodes))
    print(f"Newton matrix M = H + eps*I  (H = C diag(1/l) C^T, eps={eps:g}); "
          f"SPD, n={n_nodes}.")

    # Build the reusable solver ONCE (as the Newton loop would).
    solver = SDDMSolver(m, config=SolverConfig(tol=1e-10, maxits=2000, seed=0))
    dense = m.toarray()

    # Solve several right-hand sides (mimicking residuals -r^k across iterations),
    # all reusing the single cached factorization.
    rng = np.random.default_rng(0)
    print("\nper-RHS solve (reusing one cached factorization):")
    for k in range(4):
        r = rng.standard_normal(n_nodes)  # stand-in for the Newton residual
        t0 = time.perf_counter()
        dlam = solver.solve(r)
        dt = (time.perf_counter() - t0) * 1e3
        resid = np.linalg.norm(m @ dlam - r) / np.linalg.norm(r)
        err = np.linalg.norm(dlam - np.linalg.solve(dense, r)) / np.linalg.norm(
            np.linalg.solve(dense, r)
        )
        print(f"  rhs {k}: ||M x - b||/||b|| = {resid:.2e}, "
              f"rel-err vs dense = {err:.2e}, {dt:.3f} ms")

    # The Newton loop typically solves many OD-pair systems together: a native,
    # parallel batched solve over B right-hand sides (one cached factorization).
    batch = rng.standard_normal((32, n_nodes))
    t0 = time.perf_counter()
    sols = solver.solve_batch(batch)
    dt = (time.perf_counter() - t0) * 1e3
    max_resid = max(
        np.linalg.norm(m @ sols[i] - batch[i]) / np.linalg.norm(batch[i]) for i in range(32)
    )
    print(f"\nbatch solve (B=32, native parallel): max residual {max_resid:.2e}, {dt:.3f} ms total.")
    print("\nLaplacian solve step verified on SiouxFalls (quadratic perturbation).")


if __name__ == "__main__":
    main()
