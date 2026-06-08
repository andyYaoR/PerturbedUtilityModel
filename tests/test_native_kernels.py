"""
Parity tests for the native v0.3.0 hot-path kernels against their references.

  * ``csc_assemble_f64`` vs the numpy ``bincount`` scatter, and
  * ``recovery_poly_f64`` (the sieve polynomial inversion) vs the torch
    safeguarded-Newton ``solve_monotone`` fallback.

Both must agree to tight tolerance; they back the claim that the native path only
changes speed, not answers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import purc.static_purc as purcsolver
from purc.static_purc.backends.assembly import CSCAssembler
from purc.static_purc.perturbations._rootfind import solve_monotone
from purc.static_purc.perturbations.polynomial_sieve import PolynomialSievePerturbation
from purc.static_purc.utils.torch_compat import to_numpy

pytestmark = pytest.mark.native


def _require_native():
    if not purcsolver.native_available():
        pytest.skip("native core not built")


def test_assembly_native_vs_bincount():
    _require_native()
    rng = np.random.default_rng(0)
    inc = rng.standard_normal((8, 14))
    inc[np.abs(inc) < 0.7] = 0.0  # sparsify
    asm = CSCAssembler(inc)
    w = rng.uniform(0.1, 2.0, 14)
    eps = 0.137
    native_vals = asm.assemble_values(w, eps)  # native path
    # Reference bincount scatter (the documented fallback).
    ref = np.bincount(asm._slot, weights=asm._coeff * w[asm._srci], minlength=asm.nnz)
    ref[asm._diag_slot] += eps
    np.testing.assert_allclose(native_vals, ref, atol=1e-14)


@pytest.mark.parametrize("gamma", [[0.5], [0.4, 0.15], [0.1, 0.05, 0.02], [0.2, 0.0, 0.1, 0.05]])
def test_recovery_native_vs_rootfind(gamma):
    _require_native()
    pert = PolynomialSievePerturbation(np.array(gamma))
    g = torch.tensor(gamma, dtype=torch.float64)
    eta = torch.linspace(-0.5, 2.0, 41, dtype=torch.float64)
    lo = torch.zeros_like(eta)
    hi = torch.full_like(eta, 3.0)
    xi_native, int_native = pert._recover_native(eta, lo, hi, g)
    xi_ref, int_ref = solve_monotone(
        lambda z: pert.hprime(z, g), lambda z: pert.hsecond(z, g), eta, lo, hi
    )
    np.testing.assert_allclose(to_numpy(xi_native), to_numpy(xi_ref), atol=1e-9)
    np.testing.assert_array_equal(to_numpy(int_native), to_numpy(int_ref))


@pytest.mark.parametrize(
    "name,gamma",
    [
        ("quadratic", np.zeros(0)),
        ("entropy", np.zeros(0)),
        ("logit_entropy", np.zeros(0)),
        ("modified_entropy", np.zeros(0)),
        ("polynomial_sieve", np.array([0.3, 0.1])),
    ],
)
@pytest.mark.parametrize("mu", [1.0, 1e-3, 1e-6])
def test_barrier_recovery_native_vs_torch(name, gamma, mu):
    """The native ``recover_barrier_f64`` matches the torch root-find for every kernel."""
    _require_native()
    import scipy.sparse as sp

    import purc.static_purc.solvers.barrier as barrier
    from purc.static_purc import PUMProblem
    from purc.static_purc.constraints import GeneralPolytope
    from purc.static_purc.perturbations import get_perturbation

    inc = np.array([[1, 0, 0, 1, 0], [-1, 1, 0, 0, 1], [0, -1, 1, -1, 0], [0, 0, -1, 0, -1]], float)
    rng = np.random.default_rng(0)
    ell = rng.uniform(0.5, 2.0, inc.shape[1])
    pert = (
        get_perturbation(name, gamma=gamma)
        if name == "polynomial_sieve"
        else get_perturbation(name)
    )
    prob = PUMProblem(
        pert, GeneralPolytope(sp.csr_matrix(inc), np.array([1.0, 0.0, 0.0, -1.0]), ell=ell)
    )
    v = -rng.uniform(0.5, 3.0, inc.shape[1])
    lam = rng.standard_normal(inc.shape[0]) * 1.5

    x_native, w_native = barrier.recover_barrier_primal(prob, v, lam, gamma, mu)
    saved = barrier.native_available
    barrier.native_available = lambda: False  # force the torch fallback
    try:
        x_torch, w_torch = barrier.recover_barrier_primal(prob, v, lam, gamma, mu)
    finally:
        barrier.native_available = saved
    np.testing.assert_allclose(to_numpy(x_native), to_numpy(x_torch), atol=1e-8)
    np.testing.assert_allclose(to_numpy(w_native), to_numpy(w_torch), rtol=1e-3, atol=1e-6)


def test_ipm_backtracking_native_matches_torch_fallback():
    """
    The native fused IPM backtracking kernels equal the torch loops in solve_batch.

    Forcing the torch fallback (``native_available -> False``) must reproduce the
    native-kernel batched solve to tight tolerance: the fused safeguard kernels
    change speed, not answers.
    """
    _require_native()
    import scipy.sparse as sp

    import purc.static_purc.solvers.ipm as ipmmod
    from purc.static_purc import ForwardSolverConfig, PUMProblem
    from purc.static_purc.constraints import GeneralPolytope
    from purc.static_purc.perturbations import get_perturbation
    from purc.static_purc.solvers.ipm import IPMSolver

    rng = np.random.default_rng(3)
    n = 14
    edges = [(i, (i + 1) % n) for i in range(n)]
    for _ in range(3 * n):
        a, b = rng.integers(0, n, 2)
        if a != b:
            edges.append((int(a), int(b)))
    inc = np.zeros((n, len(edges)))
    for kk, (a, b) in enumerate(edges):
        inc[a, kk], inc[b, kk] = 1.0, -1.0
    ell = np.exp(rng.uniform(-3.0, 3.0, len(edges)))
    gamma = np.array([0.5, 0.3, 0.1])
    v = -rng.uniform(0.0, 30.0, len(edges))
    ods = []
    for _ in range(6):
        o, t = rng.choice(n, 2, replace=False)
        d = np.zeros(n)
        d[o], d[t] = 1.0, -1.0
        ods.append(d)
    b_batch = np.stack(ods)

    def _problem(d):
        pert = get_perturbation("polynomial_sieve", gamma=gamma)
        return PUMProblem(pert, GeneralPolytope(sp.csr_matrix(inc), d, ell=ell))

    s = IPMSolver(ForwardSolverConfig(max_iter=300), crossover=False, safeguard=True)
    s.preprocess(_problem(ods[0]))
    x_native = to_numpy(s.solve_batch((v, gamma), b_batch).x)

    saved = ipmmod.native_available
    ipmmod.native_available = lambda: False  # force the torch backtracking loops
    try:
        s2 = IPMSolver(ForwardSolverConfig(max_iter=300), crossover=False, safeguard=True)
        s2.preprocess(_problem(ods[0]))
        x_torch = to_numpy(s2.solve_batch((v, gamma), b_batch).x)
    finally:
        ipmmod.native_available = saved
    np.testing.assert_allclose(x_native, x_torch, atol=1e-9, rtol=0.0)
