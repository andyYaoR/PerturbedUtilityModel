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

import purcsolver
from purcsolver.backends.assembly import CSCAssembler
from purcsolver.perturbations._rootfind import solve_monotone
from purcsolver.perturbations.polynomial_sieve import PolynomialSievePerturbation
from purcsolver.utils.torch_compat import to_numpy

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
