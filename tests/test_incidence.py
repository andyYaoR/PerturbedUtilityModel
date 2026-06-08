"""
Network fast path: incidence detection and PURCLaplacianSolver routing.

  * ``detect_incidence`` recognizes node-arc incidence and rejects general A,
  * incidence problems route to PURCLaplacianSolver and yield the SAME primal
    solution as the general SDDM path (the fast path only changes how the Newton
    system is solved, not the answer),
  * an acyclic (tree) network routes to the exact O(n) forest path.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from purc.static_purc import ForwardSolverConfig, PUMProblem
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers import RegularizedSSNSolver
from purc.static_purc.utils.sparse import detect_incidence
from purc.static_purc.utils.torch_compat import to_numpy

# 4-node network with a cycle, single OD flow 0 -> 3.
CYCLE_INC = np.array(
    [[1, 0, 0, 1, 0], [-1, 1, 0, 0, 1], [0, -1, 1, -1, 0], [0, 0, -1, 0, -1]],
    dtype=float,
)
# Acyclic tree: path 0-1-2-3 plus a branch 1-4.
TREE_INC = np.array(
    [[1, 0, 0, 0], [-1, 1, 0, 1], [0, -1, 1, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
    dtype=float,
)


def test_detect_incidence_accepts_and_rejects():
    ok, edges = detect_incidence(sp.csr_matrix(CYCLE_INC))
    assert ok
    assert edges.tolist() == [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]]
    # A general matrix (a column with three nonzeros) is not incidence.
    general = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 0.0]]))
    ok2, edges2 = detect_incidence(general)
    assert not ok2 and edges2 is None


def test_incidence_auto_detected_and_routed():
    poly = GeneralPolytope(sp.csr_matrix(CYCLE_INC), np.array([1.0, 0.0, 0.0, -1.0]))
    assert poly.is_incidence
    prob = PUMProblem(get_perturbation("entropy"), poly)
    solver = RegularizedSSNSolver(ForwardSolverConfig())
    solver.preprocess(prob)
    res = solver.solve((-np.array([0.5, 0.4, 0.3, 1.1, 1.0]), torch.zeros(0)))
    assert res.success
    assert res.extras["backend_method"] in ("cholmod", "forest")


def test_purc_path_matches_general_path():
    """detect_incidence on vs off must give the same primal solution."""
    b = np.array([1.0, 0.0, 0.0, -1.0])
    ell = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    v = -np.array([0.5, 0.4, 0.3, 1.1, 1.0])
    gamma = np.array([0.3, 0.1])
    xs = {}
    for detect in (True, False):
        poly = GeneralPolytope(sp.csr_matrix(CYCLE_INC), b, ell=ell, detect_incidence=detect)
        prob = PUMProblem(get_perturbation("polynomial_sieve", gamma=gamma), poly)
        solver = RegularizedSSNSolver(ForwardSolverConfig(tol=1e-10))
        solver.preprocess(prob)
        xs[detect] = to_numpy(solver.solve((v, gamma)).x)
    np.testing.assert_allclose(xs[True], xs[False], atol=1e-8)


def test_acyclic_network_uses_forest():
    poly = GeneralPolytope(sp.csr_matrix(TREE_INC), np.array([1.0, 0.0, 0.0, -1.0, 0.0]))
    assert poly.is_incidence
    prob = PUMProblem(get_perturbation("entropy"), poly)
    solver = RegularizedSSNSolver(ForwardSolverConfig())
    solver.preprocess(prob)
    # The full tree is acyclic, so PURCLaplacianSolver routes to the forest path.
    assert solver._backend.method == "forest"
    res = solver.solve((-np.array([0.5, 0.4, 0.3, 1.0]), torch.zeros(0)))
    assert res.success
