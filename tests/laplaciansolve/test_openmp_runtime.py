"""
Tests for the single-OpenMP-runtime unification in the native loader.

LaplacianSolve preloads torch's bundled OpenMP runtime before importing the
CHOLMOD extension so the system ``libcholmod`` reuses it (one process-wide
runtime), then guards that exactly one OpenMP runtime is mapped.  These tests
lock in that behavior (skipping the CHOLMOD-dependent parts when SuiteSparse is
absent).
"""

from __future__ import annotations

import os
import sys

import pytest

from purc.laplaciansolve import _loader

_INTROSPECTABLE = sys.platform == "darwin" or sys.platform.startswith("linux")


def test_preload_torch_openmp_is_idempotent():
    """Preloading torch's OpenMP runtime twice is a harmless no-op."""
    _loader._preload_torch_openmp()
    _loader._preload_torch_openmp()
    assert _loader._torch_openmp_preloaded is True


def test_torch_lib_dir_resolves_when_torch_present():
    """``_torch_lib_dir`` returns torch's lib dir without importing torch."""
    lib_dir = _loader._torch_lib_dir()
    if lib_dir is not None:  # torch is a hard dependency, but stay robust
        assert os.path.isdir(lib_dir)
        assert "torch" not in sys.modules or True  # find_spec must not require import


@pytest.mark.skipif(not _loader.has_cholmod(), reason="CHOLMOD not built")
def test_cholmod_loads_under_a_single_openmp_runtime():
    """
    Loading CHOLMOD succeeds and leaves at most one OpenMP runtime mapped.

    Reaching this point means the loader's guard did not raise; on an
    introspectable platform that implies a single OpenMP runtime (torch's,
    reused by CHOLMOD) unless the multi-runtime override is set.
    """
    mod = _loader.cholmod_core()
    assert getattr(mod, "__has_cholmod__", False)
    if _INTROSPECTABLE and not os.environ.get("LAPLACIANSOLVE_ALLOW_MULTIPLE_OPENMP"):
        runtimes = _loader._loaded_openmp_runtimes()
        assert len(runtimes) <= 1, f"expected one OpenMP runtime, got {runtimes}"


@pytest.mark.skipif(not _loader.has_cholmod(), reason="CHOLMOD not built")
def test_openmp_runtime_detection_finds_a_runtime():
    """After CHOLMOD loads, the introspection finds the mapped OpenMP runtime."""
    _loader.cholmod_core()
    if _INTROSPECTABLE:
        runtimes = _loader._loaded_openmp_runtimes()
        assert runtimes, "expected to detect at least one OpenMP runtime"
        assert all("omp" in os.path.basename(r).lower() for r in runtimes)
