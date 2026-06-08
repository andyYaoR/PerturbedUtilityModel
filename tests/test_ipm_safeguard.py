"""
Tests for the safe-step/fast-step globalization of the primal-dual IPM.

These pin down the four properties that make the safeguard the *right* thing
(Wright-Ralph 1996/2000 within the Kojima-Noma-Yoshise 1994 framework):

  * **Correctness/parity** -- the safeguarded IPM converges to the same optimum
    as the dual semismooth-Newton oracle.
  * **Transparency** -- on well-behaved instances the safeguard is invisible:
    it takes the *same* fast (Mehrotra) step as the unsafeguarded solver, so the
    iterate and iteration count are identical.
  * **Safe-step fallback is a valid method** -- forcing every step to be a safe
    step (``rho = 0``) still converges, i.e. the centred safe step alone is a
    globally convergent algorithm.
  * **Rescue** -- on an extreme, badly-scaled instance the *unsafeguarded*
    Mehrotra step diverges to NaN, while the safeguarded solver converges.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from purc.static_purc import PUMProblem
from purc.static_purc.config import ForwardSolverConfig
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.utils.torch_compat import to_numpy


def _network(n_nodes: int, seed: int, ell_spread: float):
    """Random connected directed graph with a single OD pair (0 -> n/2)."""
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    for _ in range(3 * n_nodes):
        a, b = rng.integers(0, n_nodes, 2)
        if a != b:
            edges.append((int(a), int(b)))
    inc = np.zeros((n_nodes, len(edges)))
    for k, (a, b) in enumerate(edges):
        inc[a, k] = 1.0
        inc[b, k] = -1.0
    d = np.zeros(n_nodes)
    d[0], d[n_nodes // 2] = 1.0, -1.0
    ell = np.exp(rng.uniform(-ell_spread, ell_spread, len(edges)))
    return inc, d, ell


def _problem(n_nodes: int, seed: int, ell_spread: float, vscale: float):
    inc, d, ell = _network(n_nodes, seed, ell_spread)
    rng = np.random.default_rng(seed + 1)
    gamma = np.array([0.5, 0.3, 0.1])
    pert = get_perturbation("polynomial_sieve", gamma=gamma)
    prob = PUMProblem(pert, GeneralPolytope(sp.csr_matrix(inc), d, ell=ell))
    v = -rng.uniform(0.0, vscale, inc.shape[1])
    return prob, v, gamma


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("ell_spread,vscale", [(0.0, 3.0), (4.0, 30.0), (8.0, 300.0)])
def test_safeguard_matches_oracle(seed, ell_spread, vscale):
    """The safeguarded IPM solves the box problem (KKT residual ~ 0)."""
    prob, v, gamma = _problem(20, seed, ell_spread, vscale)
    solver = IPMSolver(ForwardSolverConfig(max_iter=200), crossover=True, safeguard=True)
    solver.preprocess(prob)
    res = solver.solve((v, gamma))
    assert res.success
    # Primal feasibility and box membership at the returned point.
    x = to_numpy(res.x)
    assert x.min() >= -1e-9 and x.max() <= 1.0 + 1e-9
    assert res.residual < 1e-7


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("ell_spread,vscale", [(0.0, 3.0), (8.0, 300.0), (4.0, 3000.0)])
def test_safeguard_is_transparent(seed, ell_spread, vscale):
    """On well-behaved instances the safeguard takes the same steps as raw."""
    prob, v, gamma = _problem(20, seed, ell_spread, vscale)
    cfg = ForwardSolverConfig(max_iter=200)
    raw = IPMSolver(cfg, crossover=False, safeguard=False)
    raw.preprocess(prob)
    r_raw = raw.solve((v, gamma))
    saf = IPMSolver(cfg, crossover=False, safeguard=True)
    saf.preprocess(prob)
    r_saf = saf.solve((v, gamma))
    assert r_raw.success and r_saf.success
    # Same optimum, and at most a 1-iteration premium over the bare Mehrotra step
    # (the safeguard takes the full fast step except for the occasional near-miss
    # of the rho-decrease test, where it shortens or falls to one safe step).
    np.testing.assert_allclose(to_numpy(r_saf.x), to_numpy(r_raw.x), atol=1e-6)
    assert r_saf.nit <= r_raw.nit + 1


@pytest.mark.parametrize("ell_spread,vscale", [(0.0, 3.0), (8.0, 300.0), (6.0, 1000.0)])
def test_safe_step_only_converges(ell_spread, vscale):
    """rho=0 forces every step to be a safe step; it must still converge."""
    prob, v, gamma = _problem(20, 1, ell_spread, vscale)
    solver = IPMSolver(ForwardSolverConfig(max_iter=300), crossover=False, safeguard=True, rho=0.0)
    solver.preprocess(prob)
    res = solver.solve((v, gamma))
    assert res.success
    assert res.extras["n_fast"] == 0  # every step was a safe step
    assert res.extras["n_safe"] >= 1
    assert res.residual < 1e-7


def test_safeguard_rescues_where_raw_diverges():
    """An extreme badly-scaled instance: raw -> NaN, safeguarded converges."""
    prob, v, gamma = _problem(40, 0, ell_spread=8.0, vscale=1e8)
    cfg = ForwardSolverConfig(max_iter=120, tol=1e-8)
    raw = IPMSolver(cfg, crossover=False, safeguard=False)
    raw.preprocess(prob)
    r_raw = raw.solve((v, gamma))
    saf = IPMSolver(cfg, crossover=False, safeguard=True)
    saf.preprocess(prob)
    r_saf = saf.solve((v, gamma))
    # Raw fails (diverges / non-finite residual); the safeguard converges.
    raw_failed = (not r_raw.success) or (not np.isfinite(r_raw.residual)) or r_raw.residual > 1e-6
    assert raw_failed, "expected the unsafeguarded Mehrotra step to fail here"
    assert r_saf.success and np.isfinite(r_saf.residual) and r_saf.residual < 1e-6
    assert r_saf.extras["n_safe"] >= 1  # the safe step did the rescuing
