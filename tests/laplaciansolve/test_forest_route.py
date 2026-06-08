"""
Stress tests for the forest-LDL fast path and the auto-select router.

Covers: exact correctness of the native forest factorization vs a dense solve on
many random forests (paths, stars, caterpillars, random spanning forests, single
nodes, disconnected components); the routing decision (forest vs cholmod vs
approxchol, including the O(1) gate and forced ``method="forest"``); numeric
refactor (update) and batched solves on the forest route; and an end-to-end PURC-
like Newton sequence whose active set shrinks to a tree, asserting the route flips
to ``"forest"`` and stays exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import random_connected_graph
from scipy import sparse

pytest.importorskip(
    "purc.laplaciansolve._laplaciansolve_core",
    reason="native core not built (run: pip install --no-build-isolation -e .)",
)

from purc.laplaciansolve import SDDMSolver, SolverConfig  # noqa: E402
from purc.laplaciansolve.forest_solver import ForestLDLSDDM  # noqa: E402
from purc.laplaciansolve.reference import lap  # noqa: E402


# --------------------------------------------------------------------------- #
# Random forest generators (all return a symmetric, positive-weight adjacency). #
# --------------------------------------------------------------------------- #
def _adj(n, edges, rng):
    """
    Build a symmetric CSC adjacency from an edge list with random positive weights.

    Args:
        n: Number of vertices.
        edges: Iterable of ``(u, v)`` pairs (u != v).
        rng: A numpy random generator for the weights.

    Returns:
        The symmetric adjacency as a CSC matrix.

    """
    rows, cols, data = [], [], []
    for u, v in edges:
        w = float(rng.uniform(0.1, 10.0))
        rows += [u, v]
        cols += [v, u]
        data += [w, w]
    return sparse.csc_matrix((data, (rows, cols)), shape=(n, n))


def _random_forest_edges(n, rng, drop_frac=0.0):
    """
    Random spanning-tree edges on ``n`` nodes (optionally dropped to a forest).

    Args:
        n: Number of vertices.
        rng: A numpy random generator.
        drop_frac: Fraction of tree edges to drop (creating multiple components).

    Returns:
        A list of ``(u, v)`` forest edges.

    """
    perm = rng.permutation(n)
    edges = []
    for i in range(1, n):
        v = int(perm[i])
        u = int(perm[rng.integers(0, i)])  # attach to an earlier node -> acyclic
        edges.append((u, v))
    if drop_frac > 0.0:
        keep = rng.random(len(edges)) >= drop_frac
        edges = [e for e, k in zip(edges, keep) if k]
    return edges


def _sddm(adj, eps):
    """
    Assemble the SDDM matrix M = lap(adj) + eps*I as sorted CSC.

    Args:
        adj: Symmetric adjacency.
        eps: Positive regularizer.

    Returns:
        The SDDM matrix (CSC, sorted indices).

    """
    m = sparse.csc_matrix(lap(adj) + eps * sparse.eye(adj.shape[0]))
    m.sort_indices()
    return m


def _dense_solve(m, b):
    """
    Dense reference solution of ``m x = b``.

    Args:
        m: SDDM matrix.
        b: Right-hand side(s).

    Returns:
        The exact dense solution.

    """
    return np.linalg.solve(m.toarray(), b.T).T


# --------------------------------------------------------------------------- #
# Exactness over many random forests.                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(25))
def test_forest_exact_random(seed):
    """Forest LDL matches a dense solve to ~machine precision on random forests."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 120))
    eps = float(10.0 ** rng.uniform(-6, -1))
    drop = float(rng.uniform(0.0, 0.4))  # often disconnected (multi-tree forest)
    adj = _adj(n, _random_forest_edges(n, rng, drop), rng)
    m = _sddm(adj, eps)

    solver = SDDMSolver(m, config=SolverConfig(method="forest", seed=seed))
    assert solver.method == "forest"
    b = rng.standard_normal(n)
    x = solver.solve(b)
    assert np.linalg.norm(m @ x - b) / np.linalg.norm(b) < 1e-9
    assert np.linalg.norm(x - _dense_solve(m, b[None, :])[0]) / (
        np.linalg.norm(_dense_solve(m, b[None, :])[0]) + 1e-300
    ) < 1e-7


@pytest.mark.parametrize("shape", ["path", "star", "caterpillar", "binary_tree"])
def test_forest_special_shapes(shape):
    """Exactness on adversarial tree shapes (deep path, wide star, etc.)."""
    rng = np.random.default_rng(hash(shape) % 2**32)
    n = 200
    if shape == "path":
        edges = [(i, i + 1) for i in range(n - 1)]
    elif shape == "star":
        edges = [(0, i) for i in range(1, n)]
    elif shape == "caterpillar":
        edges = [(i, i + 1) for i in range(n // 2 - 1)]
        edges += [(i, n // 2 + i) for i in range(n // 2)]
    else:  # binary_tree
        edges = [((i - 1) // 2, i) for i in range(1, n)]
    adj = _adj(n, edges, rng)
    m = _sddm(adj, 1e-4)
    solver = SDDMSolver(m, config=SolverConfig(method="forest"))
    assert solver.method == "forest"
    b = rng.standard_normal(n)
    x = solver.solve(b)
    assert np.linalg.norm(m @ x - b) / np.linalg.norm(b) < 1e-8


def test_single_node_and_isolated():
    """A 1x1 system and a fully diagonal (all-isolated) system solve exactly."""
    m1 = sparse.csc_matrix(np.array([[3.0]]))
    s1 = SDDMSolver(m1, config=SolverConfig(method="auto"))
    assert s1.method == "forest"
    assert abs(s1.solve(np.array([6.0]))[0] - 2.0) < 1e-12

    m = sparse.csc_matrix(np.diag([2.0, 5.0, 10.0, 4.0]))
    m.sort_indices()
    s = SDDMSolver(m, config=SolverConfig(method="auto"))
    assert s.method == "forest"
    b = np.array([2.0, 10.0, 30.0, 8.0])
    assert np.allclose(s.solve(b), [1.0, 2.0, 3.0, 2.0], atol=1e-12)


# --------------------------------------------------------------------------- #
# Routing decisions.                                                           #
# --------------------------------------------------------------------------- #
def test_router_cycle_goes_to_cholmod():
    """A cyclic (connected, dense) graph is NOT routed to forest."""
    from purc.laplaciansolve._loader import has_cholmod

    a = random_connected_graph(80, 0.08, seed=7)  # many cycles
    m = _sddm(a, 1e-3)
    s = SDDMSolver(m, config=SolverConfig(method="auto"))
    assert s.method == ("cholmod" if has_cholmod() else "approxchol")


def test_router_forced_forest_on_cycle_raises():
    """method='forest' on a cyclic matrix raises (no silent fallback)."""
    rng = np.random.default_rng(0)
    edges = [(0, 1), (1, 2), (2, 0)]  # triangle
    m = _sddm(_adj(3, edges, rng), 1e-2)
    with pytest.raises(ValueError, match="acyclic"):
        SDDMSolver(m, config=SolverConfig(method="forest"))


def test_forest_gate_rejects_dense_without_building():
    """The O(1) gate flags a clearly-cyclic sparse matrix as non-forest."""
    from purc.laplaciansolve.solver import _forest_gate

    a = random_connected_graph(60, 0.1, seed=3)
    m = _sddm(a, 1e-3)
    assert _forest_gate(m) is False  # >= n edges -> certainly cyclic
    # A path has n-1 edges -> gate must let it through.
    path = _adj(60, [(i, i + 1) for i in range(59)], np.random.default_rng(0))
    assert _forest_gate(_sddm(path, 1e-3)) is True


def test_is_forest_property_matches_truth():
    """ForestLDLSDDM.is_forest agrees with the structural edge count."""
    rng = np.random.default_rng(11)
    # Forest.
    fm = _sddm(_adj(50, _random_forest_edges(50, rng), rng), 1e-3)
    assert ForestLDLSDDM(fm).is_forest is True
    # Forest + one extra chord -> cycle.
    edges = _random_forest_edges(50, rng)
    edges.append((0, 25))  # add a chord between two existing tree nodes
    cm = _sddm(_adj(50, edges, rng), 1e-3)
    assert ForestLDLSDDM(cm).is_forest is False


# --------------------------------------------------------------------------- #
# Reuse: batch + numeric update on the forest route.                           #
# --------------------------------------------------------------------------- #
def test_forest_batch_matches_single_and_dense():
    """Batched forest solve equals per-RHS solves and the dense solution."""
    rng = np.random.default_rng(5)
    n = 90
    m = _sddm(_adj(n, _random_forest_edges(n, rng, 0.2), rng), 1e-3)
    s = SDDMSolver(m, config=SolverConfig(method="forest"))
    rhs = rng.standard_normal((16, n))
    xb = s.solve_batch(rhs)
    assert xb.shape == (16, n)
    xd = _dense_solve(m, rhs)
    for i in range(16):
        assert np.allclose(xb[i], s.solve(rhs[i]), atol=1e-9, rtol=0.0)
        assert np.linalg.norm(xb[i] - xd[i]) / np.linalg.norm(xd[i]) < 1e-7


def test_forest_update_refactors_same_pattern():
    """update() re-solves new weights/eps on the same tree without rebuilding."""
    rng = np.random.default_rng(9)
    n = 70
    edges = _random_forest_edges(n, rng, 0.1)
    s = SDDMSolver(_sddm(_adj(n, edges, rng), 1e-3), config=SolverConfig(method="forest"))
    # New weights on the SAME edges + a different eps -> same sparsity pattern.
    m2 = _sddm(_adj(n, edges, rng), 5e-2)
    s.update(m2)
    b = rng.standard_normal(n)
    x = s.solve(b)
    assert np.linalg.norm(m2 @ x - b) / np.linalg.norm(b) < 1e-9
    # A different pattern must be rejected.
    bad = _sddm(_adj(n, edges + [(edges[0][0], edges[-1][1])], rng), 1e-3)
    with pytest.raises(ValueError, match="same sparsity pattern"):
        s.update(bad)


def test_forest_torch_io():
    """Forest route accepts/returns torch tensors (CPU)."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    n = 60
    m = _sddm(_adj(n, _random_forest_edges(n, rng), rng), 1e-3)
    s = SDDMSolver(m, config=SolverConfig(method="forest"))
    b = rng.standard_normal(n)
    x = s.solve(torch.tensor(b, dtype=torch.float64))
    assert isinstance(x, torch.Tensor)
    assert np.linalg.norm(m @ x.numpy() - b) / np.linalg.norm(b) < 1e-8


# --------------------------------------------------------------------------- #
# End-to-end: a Newton-like sequence whose active set shrinks to a tree.       #
# --------------------------------------------------------------------------- #
def test_newton_sequence_flips_to_forest():
    """
    As the active edge set shrinks from cyclic to a spanning tree, the route
    flips auto -> forest and every solve stays exact against its own matrix.
    """
    from purc.laplaciansolve._loader import has_cholmod

    rng = np.random.default_rng(2)
    n = 100
    base = random_connected_graph(n, 0.06, seed=2)  # cyclic active set
    coo = sparse.triu(base, k=1).tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    rng.shuffle(edges)

    # Build a spanning tree (acyclic subset) via union-find, then peel extra edges.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree, extra = [], []
    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree.append((u, v))
        else:
            extra.append((u, v))

    saw_cycle_route, saw_forest_route = False, False
    # Sequence: full (tree+extra) -> drop extras one by one -> tree (a forest).
    for k in range(len(extra) + 1):
        active = tree + extra[: len(extra) - k]
        adj = _adj(n, active, rng)
        m = _sddm(adj, 10.0 ** rng.uniform(-6, -2))
        s = SDDMSolver(m, config=SolverConfig(method="auto", seed=2))
        b = rng.standard_normal(n)
        x = s.solve(b)
        assert np.linalg.norm(m @ x - b) / np.linalg.norm(b) < 1e-7
        if s.method == "forest":
            saw_forest_route = True
        else:
            assert s.method == ("cholmod" if has_cholmod() else "approxchol")
            saw_cycle_route = True

    assert saw_forest_route, "the spanning-tree (acyclic) step must route to forest"
    if extra:
        assert saw_cycle_route, "the cyclic steps must route to cholmod/approxchol"
