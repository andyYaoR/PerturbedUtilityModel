"""
Batched solve over OD-pairs (v0.5.0).

OD-pairs share the network ``A``, the utilities ``v`` and the perturbation, and
differ only in the demand ``b``.  ``solve_batch`` runs one vectorized Newton loop
over all systems; its primal solutions must match solving each system on its own,
and a per-system converged mask reports which finished.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from purc.static_purc import PUMProblem, SSNConfig
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers import RegularizedSSNSolver
from purc.static_purc.utils.torch_compat import to_numpy


def _bidirectional_grid(side=5):
    """A grid with two directed arcs per edge (all OD-pairs feasible)."""
    import networkx as nx

    g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    arcs = [(u, w) for u, w in g.edges()] + [(w, u) for u, w in g.edges()]
    n = g.number_of_nodes()
    rows, cols, vals = [], [], []
    for j, (u, w) in enumerate(arcs):
        rows += [u, w]
        cols += [j, j]
        vals += [1.0, -1.0]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(arcs)))
    return A, n, len(arcs)


def _demands(n, n_sys, seed=0):
    rng = np.random.default_rng(seed)
    bb = np.zeros((n_sys, n))
    for i in range(n_sys):
        o, d = rng.integers(0, n, 2)
        if o == d:
            d = (o + 1) % n
        bb[i, o] = 1.0
        bb[i, int(d)] = -1.0
    return bb


@pytest.mark.parametrize("kernel", ["entropy", "polynomial_sieve"])
def test_batch_matches_per_system_loop(kernel):
    A, n, m = _bidirectional_grid(5)
    rng = np.random.default_rng(1)
    v = -rng.uniform(0.2, 1.5, m)
    gamma = np.array([0.3, 0.1]) if kernel == "polynomial_sieve" else np.zeros(0)
    poly = GeneralPolytope(A, np.zeros(n), validate=False)
    prob = PUMProblem(
        get_perturbation(kernel, **({"gamma": gamma} if kernel == "polynomial_sieve" else {})), poly
    )
    bb = _demands(n, 12, seed=2)

    solver = RegularizedSSNSolver(SSNConfig(tol=1e-9))
    solver.preprocess(prob)
    res = solver.solve_batch((v, gamma), bb, lam0=np.zeros((12, n)))
    assert res.success
    assert res.extras["converged_mask"].all()

    x_loop = np.empty((12, m))
    for i in range(12):
        s = RegularizedSSNSolver(SSNConfig(tol=1e-9))
        s.preprocess(prob)
        x_loop[i] = to_numpy(s.solve((v, gamma), b=bb[i], lam0=np.zeros(n)).x)
    np.testing.assert_allclose(to_numpy(res.x), x_loop, atol=1e-7)


def test_batch_shapes_and_conjugate():
    A, n, m = _bidirectional_grid(4)
    rng = np.random.default_rng(3)
    v = -rng.uniform(0.2, 1.5, m)
    poly = GeneralPolytope(A, np.zeros(n), validate=False)
    prob = PUMProblem(get_perturbation("entropy"), poly)
    bb = _demands(n, 5, seed=4)
    solver = RegularizedSSNSolver(SSNConfig(tol=1e-9))
    solver.preprocess(prob)
    res = solver.solve_batch((v, torch.zeros(0)), bb)
    assert tuple(res.x.shape) == (5, m)
    assert tuple(res.lam.shape) == (5, n)
    assert tuple(res.conjugate.shape) == (5,)
    # Conjugate per system: v . x* - F(x*).
    ell = poly.ell
    expected = (torch.as_tensor(v) * res.x).sum(dim=1) - (
        ell * prob.perturbation.h(res.x, torch.zeros(0))
    ).sum(dim=1)
    np.testing.assert_allclose(to_numpy(res.conjugate), to_numpy(expected), atol=1e-9)


def test_single_solver_does_not_prematurely_stall():
    """
    Regression: a large-residual plateau must NOT trigger the stall stop.

    (The dual gradient r is non-monotone; stall only fires near the floor.)
    """
    A, n, m = _bidirectional_grid(8)
    rng = np.random.default_rng(0)
    v = -rng.uniform(0.2, 1.5, m)
    gamma = np.array([0.3, 0.1])
    poly = GeneralPolytope(A, np.zeros(n), validate=False)
    prob = PUMProblem(get_perturbation("polynomial_sieve", gamma=gamma), poly)
    bb = _demands(n, 16, seed=0)
    solver = RegularizedSSNSolver(SSNConfig(tol=1e-9))
    solver.preprocess(prob)
    for i in range(16):
        s = RegularizedSSNSolver(SSNConfig(tol=1e-9))
        s.preprocess(prob)
        res = s.solve((v, gamma), b=bb[i], lam0=np.zeros(n))
        assert res.success, f"OD-pair {i} failed to converge (residual {res.residual})"
