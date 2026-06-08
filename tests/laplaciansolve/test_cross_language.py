"""
Cross-language validation against Julia's Laplacians.jl.

Loads fixtures produced by ``scripts/gen_julia_fixtures.jl`` (saved adjacency,
right-hand side, and the ``approxchol_lap`` solution) and checks that the ported
solver reproduces the *same* system to the *same* accuracy.  Because the Julia
and Python builds consume different RNG streams, the two approximate
factorizations differ; we therefore validate functionally: identical residual
quality and close agreement of the two tol-accurate solutions (loosened by the
problem's conditioning).  The tests skip cleanly when no fixtures are present.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest
from scipy import sparse

from purc.laplaciansolve.reference import approxchol_lap, lap

pytestmark = pytest.mark.cross_language

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
_FIXTURES = sorted(glob.glob(os.path.join(_FIXTURE_DIR, "*.npz")))


def _load(path: str):
    """
    Load a fixture into ``(adjacency, b, x_julia, pcgits, tol)``.

    Args:
        path: Path to the ``.npz`` fixture.

    Returns:
        A tuple ``(a, b, x_julia, pcgits, tol)``.

    """
    data = np.load(path)
    n = int(data["n"])
    a = sparse.csc_matrix((data["data"], data["indices"], data["indptr"]), shape=(n, n))
    return a, data["b"], data["x_julia"], int(data["pcgits"]), float(data["tol"])


@pytest.mark.skipif(not _FIXTURES, reason="no Julia fixtures generated yet")
@pytest.mark.parametrize("path", _FIXTURES, ids=[os.path.basename(p) for p in _FIXTURES])
def test_matches_julia_fixture(path):
    """The ported solver matches Julia's residual and solution on each fixture."""
    a, b, x_julia, pcgits, tol = _load(path)
    la = lap(a)

    # 1) Julia's saved solution actually solves the system (fixture sanity).
    assert np.linalg.norm(la @ x_julia - b) / np.linalg.norm(b) < 10 * tol

    # 2) Our solver reaches the same tolerance on the identical matrix + RHS.
    x = approxchol_lap(a, tol=tol, maxits=4000, seed=0)(b)
    assert np.linalg.norm(la @ x - b) / np.linalg.norm(b) < 1e-6

    # 3) Two tol-accurate solutions of the same system agree (loosely).
    denom = np.linalg.norm(x_julia) or 1.0
    assert np.linalg.norm(x - x_julia) / denom < 1e-3
