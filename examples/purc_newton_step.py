r"""
PURC semismooth-Newton inner solve via the incidence-level API.

The PURC Newton system at iteration ``k`` is

    (H^k + eps_k I) d_lambda = -r^k,   H^k = C_{S^k} diag(D^k) C_{S^k}^T,

with ``C`` the node-link incidence, ``S^k`` the active links, ``D^k_i =
1/(l_i h''(x_i; gamma))`` the positive edge weights, and ``eps_k = min(eps_0,
||r^k||)`` an adaptive regularizer that makes ``M = H + eps I`` SDDM (SPD).

This example shows how the semismooth-Newton *outer* solver should drive
:class:`purc.laplaciansolve.PURCLaplacianSolver`:

* build the solver **once** from the fixed edge list (it caches the Laplacian
  pattern + the edge -> CSC maps and picks the route);
* each iteration, hand it the per-edge weights ``D`` (computed by the Newton
  solver from the perturbation Hessian), the active mask, ``eps_k``, and the RHS;
  it assembles ``M`` natively and solves - returning a result dict.

It does NOT run a full equilibrium loop; it drives a synthetic drifting-weights
sequence and a batch over several OD right-hand sides to exercise the solve step.

Run:  python examples/purc_newton_step.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse

from purc.laplaciansolve import PURCLaplacianSolver, SolverConfig
from purc.laplaciansolve.reference import lap


def grid_edges(g: int):
    """
    Return ``(n_nodes, edges)`` for a g x g 4-neighbour grid (a cyclic network).

    Args:
        g: Grid side length.

    Returns:
        The node count and an ``(m, 2)`` int array of undirected edge endpoints.

    """
    edges = []
    for i in range(g):
        for j in range(g):
            k = i * g + j
            if i + 1 < g:
                edges.append((k, k + g))
            if j + 1 < g:
                edges.append((k, k + 1))
    return g * g, np.array(edges, dtype=np.int64)


def reference_matrix(edges, n, weights, eps):
    """
    Assemble M = C diag(weights) C^T + eps*I with scipy (an independent oracle).

    Args:
        edges: ``(m, 2)`` edge endpoints.
        n: Node count.
        weights: Length-``m`` edge weights.
        eps: Scalar regularizer.

    Returns:
        The SDDM matrix as a dense array.

    """
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adj = sparse.csc_matrix((np.concatenate([weights, weights]), (rows, cols)), shape=(n, n))
    return (lap(adj) + eps * sparse.identity(n)).toarray()


def main() -> None:
    """Drive a synthetic PURC Newton-step sequence through PURCLaplacianSolver."""
    n, edges = grid_edges(8)
    m = edges.shape[0]
    lengths = np.linspace(1.0, 3.0, m)  # link lengths l_i
    solver = PURCLaplacianSolver(edges, n, config=SolverConfig(tol=1e-10))
    print(f"network: {n} nodes, {m} links;  route = {solver.method!r} (phase {solver.phase!r})")

    rng = np.random.default_rng(0)
    eps0 = 1e-8

    # Synthetic semismooth-Newton sequence: drifting iterate x_hat in (0, 1),
    # modified-entropy perturbation h''(x) = 1/(1 + x)  =>  D_i = (1 + x_i) / l_i.
    print("\nNewton-step sequence (single RHS, reusing one cached pattern):")
    x_hat = rng.uniform(0.1, 0.9, m)
    for it in range(5):
        x_hat = np.clip(x_hat + 0.05 * rng.standard_normal(m), 0.01, 0.99)
        weights = (1.0 + x_hat) / lengths  # D_i = (1 + x_i)/l_i
        active = ((x_hat > 0.02) & (x_hat < 0.98)).astype(float)  # active link mask S^k
        r = rng.standard_normal(n)  # stand-in Newton residual r^k
        eps_k = min(eps0, float(np.linalg.norm(r)))

        t0 = time.perf_counter()
        result = solver.solve_step(weights, active_mask=active, eps=eps_k, rhs=-r)
        dt = (time.perf_counter() - t0) * 1e3

        d_lambda = result["solution"]
        mref = reference_matrix(edges, n, weights * active, eps_k)
        resid = np.linalg.norm(mref @ d_lambda + r) / np.linalg.norm(r)
        print(
            f"  iter {it}: ||M d_lambda + r||/||r|| = {resid:.2e}, "
            f"phase={result['phase']}, {dt:.3f} ms"
        )
        # A direct solve is backward stable: the residual scales with the
        # conditioning kappa(M) ~ ||M||/eps_k, so at eps_k ~ 1e-8 expect ~1e-8.
        assert resid < 1e-6

    # The Newton loop solves many OD right-hand sides per iteration: one native,
    # parallel batched solve sharing the iteration's matrix (here, one weight set).
    weights = (1.0 + x_hat) / lengths
    batch = rng.standard_normal((64, n))
    t0 = time.perf_counter()
    out = solver.solve_batch(weights, eps=1e-8, rhs=batch)
    dt = (time.perf_counter() - t0) * 1e3
    mref = reference_matrix(edges, n, weights, 1e-8)
    worst = max(
        np.linalg.norm(mref @ out["solution"][i] - batch[i]) / np.linalg.norm(batch[i])
        for i in range(64)
    )
    print(f"\nbatch solve (B=64 OD RHS, shared matrix): max residual {worst:.2e}, {dt:.3f} ms total.")

    # Per-destination batch: each system has its OWN weights/active set (the
    # non-quadratic / Markovian case) - B distinct matrices, one native call.
    weights_B = np.stack([(1.0 + np.clip(x_hat + 0.1 * d, 0.01, 0.99)) / lengths for d in range(4)])
    rhs_B = rng.standard_normal((4, n))
    out = solver.solve_batch(weights_B, eps=1e-8, rhs=rhs_B)
    worst = 0.0
    for d in range(4):
        mref = reference_matrix(edges, n, weights_B[d], 1e-8)
        worst = max(worst, np.linalg.norm(mref @ out["solution"][d] - rhs_B[d]) / np.linalg.norm(rhs_B[d]))
    print(f"per-system batch (B=4 distinct matrices): max residual {worst:.2e}.")

    # An acyclic network routes to the exact zero-fill forest path: a spanning
    # tree of this grid solves via 'forest'.
    tree = PURCLaplacianSolver(_spanning_tree(edges, n), n)
    print(f"\nspanning-tree network: route = {tree.method!r} (phase {tree.phase!r}).")
    assert tree.phase == "forest"
    print("\nPURC Newton-step solve verified (assembly + routing + batch).")


def _spanning_tree(edges, n):
    """Return edges of a spanning forest of the graph (union-find), as (k, 2)."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    kept = []
    for u, v in edges:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
            kept.append((int(u), int(v)))
    return np.array(kept, dtype=np.int64)


if __name__ == "__main__":
    main()
