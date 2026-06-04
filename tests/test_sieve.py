"""
Tests for the polynomial sieve (the paper's model) and the symbolic compiler
(torch-native).

Covers Bernstein convexity, sieve recovery, sieve solves vs the independent scipy
dual oracle, and the symbolic compiler (auto-derivatives, closed-form inverse vs
root-find).  The compiled-kernel callables are NumPy; sieve/solver outputs are
torch and are bridged with ``to_numpy`` for comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import sympy as sp_sym
import torch

from purcsolver import PUMProblem, SSNConfig
from purcsolver.constraints import GeneralPolytope
from purcsolver.oracle import solve_scipy
from purcsolver.perturbations import get_perturbation
from purcsolver.perturbations._bernstein import bernstein_coeffs, is_convex
from purcsolver.perturbations.compiler import SymbolicPerturbation, compile_kernel
from purcsolver.perturbations.polynomial_sieve import PolynomialSievePerturbation
from purcsolver.solvers import RegularizedSSNSolver
from purcsolver.utils.torch_compat import to_numpy

FEASIBLE_GAMMAS = [[0.5], [0.3, 0.2], [0.1, 0.05, 0.02], [0.2, 0.0, 0.1, 0.05], [-0.4]]
INFEASIBLE_GAMMAS = [[-2.0], [0.0, -3.0]]


def _sieve_h_second(gamma, xi):
    out = np.ones_like(xi)
    for m, g in enumerate(gamma):
        out = out + g * (m + 2) * xi ** (m + 1)
    return out


@pytest.mark.parametrize("gamma", FEASIBLE_GAMMAS)
def test_bernstein_feasible_matches_positive_hessian(gamma):
    assert is_convex(np.array(gamma))
    grid = np.linspace(1e-6, 1 - 1e-6, 500)
    assert np.all(_sieve_h_second(gamma, grid) > 0)


@pytest.mark.parametrize("gamma", INFEASIBLE_GAMMAS)
def test_bernstein_infeasible_detects_nonconvexity(gamma):
    assert not is_convex(np.array(gamma))
    assert np.min(bernstein_coeffs(np.array(gamma))) < 0


@pytest.mark.parametrize("gamma", FEASIBLE_GAMMAS)
def test_sieve_inverse_relation(gamma):
    pert = PolynomialSievePerturbation(np.array(gamma))
    g = torch.tensor(gamma, dtype=torch.float64)
    h_prime_1 = 1.0 + sum(gamma)
    eta = torch.linspace(0.02, 0.9 * h_prime_1, 17, dtype=torch.float64)
    lo = torch.zeros_like(eta)
    hi = torch.ones_like(eta)
    xi_star, interior = pert.primal_recovery(eta, lo, hi, g)
    assert bool(interior.all())
    np.testing.assert_allclose(to_numpy(pert.hprime(xi_star, g)), to_numpy(eta), atol=1e-10)


@pytest.mark.parametrize("gamma", FEASIBLE_GAMMAS)
def test_sieve_gamma_feasible_flag(gamma):
    assert PolynomialSievePerturbation().gamma_feasible(np.array(gamma))


@pytest.mark.parametrize("gamma", INFEASIBLE_GAMMAS)
def test_sieve_gamma_infeasible_flag(gamma):
    assert not PolynomialSievePerturbation().gamma_feasible(np.array(gamma))


def _network():
    inc = np.array(
        [[1, 0, 0, 1, 0], [-1, 1, 0, 0, 1], [0, -1, 1, -1, 0], [0, 0, -1, 0, -1]],
        dtype=float,
    )
    ell = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    v = -np.array([0.5, 0.4, 0.3, 1.1, 1.0])
    return GeneralPolytope(sp.csr_matrix(inc), np.array([1.0, 0.0, 0.0, -1.0]), ell=ell), v


def _simplex(n=6, seed=3):
    rng = np.random.default_rng(seed)
    return GeneralPolytope(sp.csr_matrix(np.ones((1, n))), np.array([1.0])), rng.standard_normal(n)


@pytest.mark.parametrize("gamma", [[0.4, 0.15], [0.3, 0.1, 0.05], [0.2, 0.1, 0.0, 0.05]])
@pytest.mark.parametrize("geometry", ["network", "simplex"])
def test_sieve_solver_matches_scipy_oracle(gamma, geometry):
    poly, v = _network() if geometry == "network" else _simplex()
    gamma = np.array(gamma)
    prob = PUMProblem(get_perturbation("polynomial_sieve", gamma=gamma), poly)
    solver = RegularizedSSNSolver(SSNConfig(tol=1e-10))
    solver.preprocess(prob)
    res = solver.solve((v, gamma))
    assert res.success
    x_oracle = solve_scipy(prob, (v, gamma))
    np.testing.assert_allclose(to_numpy(res.x), x_oracle, atol=1e-7)


# -- symbolic compiler --------------------------------------------------------


@pytest.mark.parametrize("gamma", [[0.5], [0.4, 0.15], [0.1, 0.05, 0.02]])
def test_compiler_derivatives_match_analytic(gamma):
    xi = sp_sym.Symbol("xi", real=True)
    expr = xi**2 / 2 + sum(g * xi ** (m + 3) / (m + 3) for m, g in enumerate(gamma))
    ck = compile_kernel(expr, xi)
    sieve = PolynomialSievePerturbation(np.array(gamma))
    x = torch.linspace(0.01, 0.99, 21, dtype=torch.float64)
    xn = to_numpy(x)
    np.testing.assert_allclose(ck.h(xn), to_numpy(sieve.h(x, None)), atol=1e-12)
    np.testing.assert_allclose(ck.hp(xn), to_numpy(sieve.hprime(x, None)), atol=1e-12)
    np.testing.assert_allclose(ck.hpp(xn), to_numpy(sieve.hsecond(x, None)), atol=1e-12)
    assert ck.convex_on_box is True
    assert ck.degree == len(gamma) + 1


@pytest.mark.parametrize("gamma", [[0.5], [0.4, 0.15], [0.1, 0.05, 0.02]])
def test_compiler_closed_form_inverse_matches_rootfind(gamma):
    """Deg <= 4: SymPy gives radicals; they must match the safeguarded Newton."""
    xi = sp_sym.Symbol("xi", real=True)
    expr = xi**2 / 2 + sum(g * xi ** (m + 3) / (m + 3) for m, g in enumerate(gamma))
    ck = compile_kernel(expr, xi)
    assert ck.has_closed_form
    sieve = PolynomialSievePerturbation(np.array(gamma))
    eta = np.linspace(0.02, 2.0, 23)
    lo = np.zeros_like(eta)
    hi = np.full_like(eta, 3.0)
    xi_cf, _ = ck.inverse(eta, lo, hi)
    xi_rf, _ = sieve.primal_recovery(eta, lo, hi, torch.tensor(gamma, dtype=torch.float64))
    np.testing.assert_allclose(xi_cf, to_numpy(xi_rf), atol=1e-9)


def test_symbolic_perturbation_closed_vs_rootfind_agree():
    xi = sp_sym.Symbol("xi", real=True)
    expr = xi**2 / 2 + 0.4 * xi**3 / 3 + 0.15 * xi**4 / 4
    auto = SymbolicPerturbation(expr, xi, method="auto")
    rf = SymbolicPerturbation(expr, xi, method="rootfind")
    assert auto.has_closed_form_recovery
    eta = torch.linspace(0.02, 2.0, 15, dtype=torch.float64)
    lo, hi = torch.zeros_like(eta), torch.full_like(eta, 3.0)
    xa, _ = auto.primal_recovery(eta, lo, hi, None)
    xr, _ = rf.primal_recovery(eta, lo, hi, None)
    np.testing.assert_allclose(to_numpy(xa), to_numpy(xr), atol=1e-9)


def test_symbolic_perturbation_solves_like_sieve():
    """A compiled symbolic kernel solves the forward problem like the analytic one."""
    xi = sp_sym.Symbol("xi", real=True)
    gamma = [0.4, 0.15]
    expr = xi**2 / 2 + gamma[0] * xi**3 / 3 + gamma[1] * xi**4 / 4
    poly, v = _network()
    res_sym = _solve(PUMProblem(SymbolicPerturbation(expr, xi), poly), v, gamma=torch.zeros(0))
    res_sieve = _solve(
        PUMProblem(get_perturbation("polynomial_sieve", gamma=np.array(gamma)), poly),
        v,
        gamma=np.array(gamma),
    )
    np.testing.assert_allclose(to_numpy(res_sym.x), to_numpy(res_sieve.x), atol=1e-8)


def _solve(prob, v, gamma):
    solver = RegularizedSSNSolver(SSNConfig(tol=1e-10))
    solver.preprocess(prob)
    return solver.solve((v, gamma))
