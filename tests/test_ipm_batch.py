"""
Tests for the batched interior-point solver ``IPMSolver.solve_batch``.

The batched IPM vectorizes Algorithm 1 over OD-pairs that share the network,
utilities, and perturbation, differing only in the demand ``b``.  These tests pin
down the properties that make the batch trustworthy:

  * **Parity** -- the batched solve reproduces the per-OD ``IPMSolver.solve``
    to machine precision (it *is* the same algorithm, run elementwise), in the
    benign, stiff, and crossover regimes.
  * **Oracle agreement** -- on moderate instances the batched primal matches the
    independent scipy dual oracle.
  * **Per-system bookkeeping** -- the converged mask, per-system residuals, and
    per-OD fast/safe step tallies are returned with the right shapes.
  * **Warm starts** -- passing the previous multipliers cuts iterations.
  * **Unsafeguarded** -- the bare batched Mehrotra step also vectorizes.

The batched IPM and per-OD IPM share the same native, GIL-released batched
Laplacian solve, so parity here certifies the vectorization, not the linear
algebra (already covered elsewhere).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from purc.static_purc import PUMProblem
from purc.static_purc.config import ForwardSolverConfig
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.oracle import solve_scipy
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.utils.torch_compat import to_numpy

GAMMA = np.array([0.5, 0.3, 0.1])


def _network(n_nodes: int, seed: int, ell_spread: float):
    """Random connected directed graph with exponential link weights."""
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    for _ in range(3 * n_nodes):
        a, b = rng.integers(0, n_nodes, 2)
        if a != b:
            edges.append((int(a), int(b)))
    inc = np.zeros((n_nodes, len(edges)))
    for k, (a, b) in enumerate(edges):
        inc[a, k], inc[b, k] = 1.0, -1.0
    ell = np.exp(rng.uniform(-ell_spread, ell_spread, len(edges)))
    return inc, ell


def _ods(n_nodes: int, n_od: int, seed: int):
    """A list of single-unit OD demand vectors on ``n_nodes`` nodes."""
    rng = np.random.default_rng(seed + 100)
    ods = []
    for _ in range(n_od):
        o, t = rng.choice(n_nodes, 2, replace=False)
        d = np.zeros(n_nodes)
        d[o], d[t] = 1.0, -1.0
        ods.append(d)
    return ods


def _problem(inc, ell, d):
    pert = get_perturbation("polynomial_sieve", gamma=GAMMA)
    return PUMProblem(pert, GeneralPolytope(sp.csr_matrix(inc), d, ell=ell))


@pytest.mark.parametrize("crossover", [False, True])
@pytest.mark.parametrize("ell_spread,vscale", [(0.0, 3.0), (4.0, 30.0), (8.0, 3000.0)])
def test_batch_matches_per_od(crossover, ell_spread, vscale):
    """The batched IPM reproduces per-OD ``solve`` to machine precision."""
    inc, ell = _network(16, 0, ell_spread)
    n = inc.shape[0]
    ods = _ods(n, 6, 1)
    rng = np.random.default_rng(7)
    v = -rng.uniform(0.0, vscale, inc.shape[1])
    b_batch = np.stack(ods)
    cfg = ForwardSolverConfig(max_iter=300)

    prob = _problem(inc, ell, ods[0])
    batched = IPMSolver(cfg, crossover=crossover, safeguard=True)
    batched.preprocess(prob)
    rb = batched.solve_batch((v, GAMMA), b_batch)
    xb = to_numpy(rb.x)
    assert rb.success
    assert xb.shape == (len(ods), inc.shape[1])

    for i, d in enumerate(ods):
        solo = IPMSolver(cfg, crossover=crossover, safeguard=True)
        solo.preprocess(_problem(inc, ell, d))
        xi = to_numpy(solo.solve((v, GAMMA), b=d).x)
        # The batched and per-OD solvers run the identical algorithm.
        np.testing.assert_allclose(xb[i], xi, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("ell_spread,vscale", [(0.0, 3.0), (3.0, 30.0)])
def test_batch_matches_scipy_oracle(ell_spread, vscale):
    """On moderate instances the batched primal matches the scipy dual oracle."""
    inc, ell = _network(16, 2, ell_spread)
    n = inc.shape[0]
    ods = _ods(n, 5, 3)
    rng = np.random.default_rng(11)
    v = -rng.uniform(0.0, vscale, inc.shape[1])
    cfg = ForwardSolverConfig(max_iter=300)

    batched = IPMSolver(cfg, crossover=True, safeguard=True)
    batched.preprocess(_problem(inc, ell, ods[0]))
    xb = to_numpy(batched.solve_batch((v, GAMMA), np.stack(ods)).x)

    for i, d in enumerate(ods):
        xo = solve_scipy(_problem(inc, ell, d), (v, GAMMA))
        np.testing.assert_allclose(xb[i], xo, atol=1e-6, rtol=0.0)


def _cvxpy_sieve(A, ell, v, d):
    """Solve one sieve OD subproblem with CVXPY/Clarabel (a known external solver)."""
    import cvxpy as cp

    x = cp.Variable(A.shape[1])
    powers = 0.5 * cp.square(x)
    for j, g in enumerate(GAMMA):
        powers = powers + g / (j + 3) * cp.power(x, j + 3)
    prob = cp.Problem(cp.Minimize(ell @ powers - v @ x), [A @ x == d, x >= 0, x <= 1])
    prob.solve(solver=cp.CLARABEL)
    return x.value, prob.status


@pytest.mark.oracle
@pytest.mark.parametrize("ell_spread,vscale", [(0.0, 3.0), (3.0, 30.0)])
def test_batch_matches_cvxpy_clarabel(ell_spread, vscale):
    """The batched IPM matches a known external solver (CVXPY/Clarabel) on the sieve."""
    pytest.importorskip("cvxpy")
    inc, ell = _network(16, 21, ell_spread)
    n = inc.shape[0]
    ods = _ods(n, 5, 23)
    v = -np.random.default_rng(29).uniform(0.0, vscale, inc.shape[1])
    prob = _problem(inc, ell, ods[0])
    batched = IPMSolver(ForwardSolverConfig(max_iter=300), crossover=True, safeguard=True)
    batched.preprocess(prob)
    xb = to_numpy(batched.solve_batch((v, GAMMA), np.stack(ods)).x)
    A = prob.constraint.A
    for i, d in enumerate(ods):
        xc, status = _cvxpy_sieve(A, to_numpy(prob.constraint.ell), v, d)
        assert xc is not None and "optimal" in str(status)
        np.testing.assert_allclose(xb[i], xc, atol=1e-5, rtol=0.0)


def test_batch_extras_and_converged_mask():
    """Per-system residuals, converged mask, and step tallies have right shapes."""
    inc, ell = _network(14, 4, 4.0)
    ods = _ods(inc.shape[0], 5, 5)
    v = -np.random.default_rng(13).uniform(0.0, 30.0, inc.shape[1])
    batched = IPMSolver(ForwardSolverConfig(max_iter=300), crossover=False, safeguard=True)
    batched.preprocess(_problem(inc, ell, ods[0]))
    r = batched.solve_batch((v, GAMMA), np.stack(ods))
    assert r.extras["n_systems"] == len(ods)
    assert to_numpy(r.extras["converged_mask"]).shape == (len(ods),)
    assert bool(to_numpy(r.extras["converged_mask"]).all())
    assert to_numpy(r.extras["per_system_residual"]).shape == (len(ods),)
    assert to_numpy(r.extras["n_fast"]).shape == (len(ods),)
    assert to_numpy(r.extras["n_safe"]).shape == (len(ods),)
    # Every per-OD step is a fast or a safe step.
    assert int(to_numpy(r.extras["n_fast"]).sum() + to_numpy(r.extras["n_safe"]).sum()) > 0


def test_batch_warm_start_reduces_iterations():
    """Warm-starting the multipliers across a theta-update cuts IPM iterations."""
    inc, ell = _network(18, 6, 4.0)
    ods = _ods(inc.shape[0], 6, 7)
    b_batch = np.stack(ods)
    v = -np.random.default_rng(17).uniform(0.0, 30.0, inc.shape[1])
    cfg = ForwardSolverConfig(max_iter=300)
    solver = IPMSolver(cfg, crossover=False, safeguard=True)
    solver.preprocess(_problem(inc, ell, ods[0]))

    cold = solver.solve_batch((v, GAMMA), b_batch, lam0=np.zeros_like(b_batch))
    warm_lam = to_numpy(cold.lam)
    # Re-solve at a slightly perturbed utility, warm-started at the previous lambda.
    v2 = v * 1.01
    warm = solver.solve_batch((v2, GAMMA), b_batch, lam0=warm_lam)
    cold2 = solver.solve_batch((v2, GAMMA), b_batch, lam0=np.zeros_like(b_batch))
    assert warm.success and cold2.success
    assert warm.nit <= cold2.nit


def test_unsafeguarded_batch_converges():
    """The bare (unsafeguarded) batched Mehrotra step also vectorizes and solves."""
    inc, ell = _network(14, 8, 2.0)
    ods = _ods(inc.shape[0], 5, 9)
    v = -np.random.default_rng(19).uniform(0.0, 30.0, inc.shape[1])
    batched = IPMSolver(ForwardSolverConfig(max_iter=300), crossover=False, safeguard=False)
    batched.preprocess(_problem(inc, ell, ods[0]))
    r = batched.solve_batch((v, GAMMA), np.stack(ods))
    assert r.success and r.residual < 1e-6


def _multigraph(n_nodes: int, seed: int):
    """
    A connected directed multigraph with guaranteed anti-parallel (two-way) arcs,
    so the constraint routes to the general SDDM batch path, not the incidence path.
    """
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    edges += [((i + 1) % n_nodes, i) for i in range(n_nodes)]  # anti-parallel back-arcs
    for _ in range(2 * n_nodes):
        a, b = rng.integers(0, n_nodes, 2)
        if a != b:
            edges.append((int(a), int(b)))
    inc = np.zeros((n_nodes, len(edges)))
    for k, (a, b) in enumerate(edges):
        inc[a, k], inc[b, k] = 1.0, -1.0
    ell = np.exp(rng.uniform(-2.0, 2.0, len(edges)))
    return inc, ell


def test_general_sddm_batch_matches_per_od():
    """
    The native general-A (SDDM) batch path reproduces per-OD solves to machine eps.

    A two-way multigraph forces the general ``BatchedSDDMSolver`` route (not the
    incidence fast path); the batched values assemble at the shared CSC pattern.
    """
    inc, ell = _multigraph(15, 5)
    ods = _ods(inc.shape[0], 6, 4)
    v = -np.random.default_rng(13).uniform(0.0, 30.0, inc.shape[1])
    cfg = ForwardSolverConfig(max_iter=300)
    batched = IPMSolver(cfg, crossover=False, safeguard=True)
    batched.preprocess(_problem(inc, ell, ods[0]))
    assert batched._backend.kind == "sddm"  # the general path, exercised on purpose
    xb = to_numpy(batched.solve_batch((v, GAMMA), np.stack(ods)).x)
    for i, d in enumerate(ods):
        solo = IPMSolver(cfg, crossover=False, safeguard=True)
        solo.preprocess(_problem(inc, ell, d))
        xi = to_numpy(solo.solve((v, GAMMA), b=d).x)
        np.testing.assert_allclose(xb[i], xi, atol=1e-9, rtol=0.0)


def test_batched_assembly_matches_per_system():
    """``assemble_values_batch`` equals stacked per-system ``assemble_values``."""
    from purc.static_purc.backends.assembly import CSCAssembler

    inc, _ = _multigraph(12, 1)
    asm = CSCAssembler(sp.csr_matrix(inc))
    rng = np.random.default_rng(2)
    w = rng.uniform(0.1, 3.0, (7, inc.shape[1]))
    eps = rng.uniform(1e-6, 1e-3, 7)
    batch = asm.assemble_values_batch(w, eps)
    per = np.stack([asm.assemble_values(w[b], float(eps[b])) for b in range(7)])
    np.testing.assert_allclose(batch, per, atol=1e-12, rtol=0.0)
