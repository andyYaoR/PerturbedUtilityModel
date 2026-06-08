"""
Smoke tests for the native C++ core module (Stage 1a).

Validates that the compiled nanobind extension imports and that the zero-copy
ndarray + GIL-release calling convention works.  Skips cleanly when the
extension has not been built (e.g. a pure-Python checkout).
"""

from __future__ import annotations

import numpy as np
import pytest

core = pytest.importorskip(
    "purc.laplaciansolve._laplaciansolve_core",
    reason="native core not built (run: pip install --no-build-isolation -e .)",
)


def test_core_version():
    """The native module exposes its version string."""
    assert isinstance(core.__core_version__, str)
    assert core.__core_version__


def test_axpy_zero_copy_inplace():
    """axpy_f64 updates the output array in place via the zero-copy contract."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    core.axpy_f64(2.0, x, y)
    assert np.array_equal(y, [12.0, 24.0, 36.0, 48.0])


def test_axpy_empty():
    """axpy_f64 handles a zero-length array without error."""
    x = np.zeros(0, dtype=np.float64)
    y = np.zeros(0, dtype=np.float64)
    core.axpy_f64(1.0, x, y)
    assert y.shape == (0,)
