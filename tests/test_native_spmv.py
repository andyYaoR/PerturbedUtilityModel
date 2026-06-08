"""
Native CSR SpMV parity tests.

The hot-path matvec ``A @ x`` (and ``A^T @ lambda``) uses a GIL-released native
float64 kernel when the core is built.  These tests assert it matches SciPy to
floating-point precision (``-march=native`` enables FMA, which breaks bit-exact
equality) and that the polytope's ``matvec`` / ``rmatvec`` agree with the dense
product, on rectangular and rank-deficient matrices.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import purc.static_purc as purcsolver
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.utils.spmv import CSRMatVec
from purc.static_purc.utils.torch_compat import to_numpy


@pytest.mark.parametrize("shape", [(50, 120), (120, 50), (1, 7), (30, 30)])
def test_csrmatvec_matches_scipy(shape):
    rng = np.random.default_rng(0)
    A = sp.random(*shape, density=0.08, random_state=1, format="csr")
    A.data = rng.standard_normal(A.nnz)
    x = rng.standard_normal(shape[1])
    mv = CSRMatVec(A)
    np.testing.assert_allclose(mv.matvec(x), A @ x, rtol=1e-9, atol=1e-12)


@pytest.mark.native
def test_native_csr_spmv_kernel_matches_scipy():
    if not purcsolver.native_available():
        pytest.skip("native core not built")
    from purc.static_purc.utils.native import native_core

    rng = np.random.default_rng(1)
    A = sp.random(80, 200, density=0.05, random_state=2, format="csr")
    A.data = rng.standard_normal(A.nnz)
    A.sort_indices()
    x = np.ascontiguousarray(rng.standard_normal(200))
    y = np.empty(80)
    native_core().csr_spmv_f64(
        A.indptr.astype(np.int64),
        A.indices.astype(np.int64),
        np.ascontiguousarray(A.data),
        x,
        y,
    )
    np.testing.assert_allclose(y, A @ x, rtol=1e-9, atol=1e-12)


def test_polytope_matvec_rmatvec_match_dense():
    inc = np.array(
        [[1, 0, 0, 1, 0], [-1, 1, 0, 0, 1], [0, -1, 1, -1, 0], [0, 0, -1, 0, -1]],
        dtype=float,
    )
    poly = GeneralPolytope(sp.csr_matrix(inc), np.array([1.0, 0.0, 0.0, -1.0]))
    x = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)
    lam = torch.tensor([0.3, -0.2, 0.5, 0.1], dtype=torch.float64)
    np.testing.assert_allclose(to_numpy(poly.matvec(x)), inc @ to_numpy(x), atol=1e-14)
    np.testing.assert_allclose(to_numpy(poly.rmatvec(lam)), inc.T @ to_numpy(lam), atol=1e-14)
