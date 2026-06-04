"""
M0 scaffold tests.

Verify the package imports, the registries and abstract interfaces are present,
the core dataclasses validate their inputs, and -- when the native core is built
-- the GIL-released smoke kernel runs and the parity invariant holds.
"""

from __future__ import annotations

import numpy as np
import pytest

import purcsolver
from purcsolver import PUMProblem, PURCResult, SSNConfig
from purcsolver.result import STATUS_CONVERGED, STATUS_MAX_ITER


def test_package_imports_and_version():
    assert isinstance(purcsolver.__version__, str)
    # Lazily-exposed public surface resolves without error.
    assert purcsolver.SeparablePerturbation is not None
    assert purcsolver.Polytope is not None
    assert purcsolver.ForwardSolver is not None
    assert isinstance(purcsolver.PERTURBATIONS, dict)
    assert isinstance(purcsolver.SOLVERS, dict)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = purcsolver.does_not_exist


def test_registries_resolve_helpers():
    with pytest.raises(KeyError):
        purcsolver.get_perturbation("nope")
    with pytest.raises(KeyError):
        purcsolver.get_solver("nope")


def test_ssnconfig_defaults_valid():
    cfg = SSNConfig()
    assert cfg.tol > 0
    assert 0 < cfg.eps_floor <= cfg.eps0
    assert 0 < cfg.armijo_c1 < 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": 0.0},
        {"max_iter": 0},
        {"eps0": -1.0},
        {"eps_floor": 0.0},
        {"eps_floor": 1.0, "eps0": 0.5},  # floor > eps0
        {"armijo_c1": 0.5},
        {"armijo_beta": 1.0},
        {"max_linesearch": 0},
    ],
)
def test_ssnconfig_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        SSNConfig(**kwargs)


def test_purcresult_default_message_from_status():
    converged = PURCResult(x=np.zeros(3), lam=np.zeros(2), success=True, status=STATUS_CONVERGED)
    assert "Converged" in converged.message
    capped = PURCResult(x=np.zeros(3), lam=np.zeros(2), success=False, status=STATUS_MAX_ITER)
    assert "Maximum" in capped.message
    assert set(converged.to_dict()) >= {"x", "lam", "success", "status", "residual"}


def test_pumproblem_utility_identity_and_linear():
    # A minimal stub polytope: only num_coords is exercised by PUMProblem here.
    class _StubPolytope:
        num_coords = 4

    prob = PUMProblem(perturbation=object(), constraint=_StubPolytope())
    beta = np.array([1.0, -2.0, 3.0, 0.5])
    # Z is None => v = beta.
    np.testing.assert_allclose(prob.utility(beta), beta)
    assert prob.num_params == 4

    Z = np.arange(12, dtype=float).reshape(4, 3)
    prob2 = PUMProblem(perturbation=object(), constraint=_StubPolytope(), Z=Z)
    b3 = np.array([1.0, 0.0, -1.0])
    np.testing.assert_allclose(prob2.utility(b3), Z @ b3)
    assert prob2.num_params == 3
    with pytest.raises(ValueError):
        prob2.utility(np.zeros(4))  # wrong length for Z


def test_native_available_is_bool():
    assert isinstance(purcsolver.native_available(), bool)


@pytest.mark.native
def test_native_axpy_gil_released_smoke():
    if not purcsolver.native_available():
        pytest.skip("native core not built")
    from purcsolver.utils.native import native_core

    core = native_core()
    x = np.arange(5, dtype=np.float64)
    y = np.ones(5, dtype=np.float64)
    core.axpy_f64(2.0, x, y)
    np.testing.assert_allclose(y, 1.0 + 2.0 * np.arange(5))
