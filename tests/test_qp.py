"""
Correctness tests for the trust-region box+linear QP (torch reference + native kernel).

The QP ``min 1/2 d'B d + g'd s.t. lo<=d<=hi, A d>=a`` is the load-bearing inner solve
of the trust-region BFGS optimizer.  We check, over wide random batteries:

  * **KKT optimality** of the torch reference (feasibility, stationarity via the
    returned multipliers, ``mu >= 0``, complementarity);
  * **agreement with cvxpy/CLARABEL** on the objective value;
  * **edge cases** -- 1-D, no linear rows, redundant rows, tiny-norm rows, a
    constraint tight at ``d=0``, an active bound at the optimum;
  * **native vs torch parity** -- the C++ ``qp_box_linear_f64`` reproduces the
    reference to ``~1e-11`` (it only changes speed, not the answer).
"""

from __future__ import annotations

import numpy as np
import pytest

import purc.static_purc as purcsolver
from purc.estimators.debiased_fy.qp import solve_box_linear_qp, solve_box_linear_qp_torch


def _assemble(P, lo, hi, A, a):
    rows = [np.eye(P), -np.eye(P)]
    rhs = [lo, -hi]
    if A is not None and len(A):
        rows.append(A)
        rhs.append(a)
    return np.vstack(rows), np.concatenate(rhs)


def _kkt_violation(B, g, lo, hi, A, a, res):
    """Max violation of (feasibility, stationarity, mu>=0, complementarity)."""
    P = g.size
    d = res.d.numpy()
    mu = res.multipliers.numpy()
    C, b = _assemble(P, lo, hi, A, a)
    feas = float((C @ d - b).min())
    stat = float(np.abs(B @ d + g - C.T @ mu).max())
    comp = float(np.abs(mu * (C @ d - b)).max())
    return max(stat, comp, max(0.0, -feas), max(0.0, -float(mu.min())))


def _rand_qp(rng, P=None, degenerate=False):
    P = P or int(rng.integers(1, 9))
    Qm, _ = np.linalg.qr(rng.standard_normal((P, P)))
    eig = np.exp(np.linspace(0, np.log(10 ** rng.uniform(0, 5)), P))
    B = (Qm * eig) @ Qm.T
    B = 0.5 * (B + B.T)
    g = rng.standard_normal(P) * rng.uniform(0.1, 8)
    delta = rng.uniform(0.05, 3.0)
    lo, hi = -delta * np.ones(P), delta * np.ones(P)
    m = int(rng.integers(0, 8))
    A = a = None
    if m:
        A = rng.standard_normal((m, P))
        a = -np.abs(rng.standard_normal(m)) * rng.uniform(0, 2)
        if degenerate:
            if rng.random() < 0.4:
                A[rng.integers(m)] *= rng.uniform(1e-5, 1e-3)        # tiny-norm row
            if m >= 2 and rng.random() < 0.5:  # redundant row
                j = int(rng.integers(1, m))
                A[j], a[j] = A[0], a[0]
            if m >= 3 and rng.random() < 0.4:
                a[int(rng.integers(m))] = 0.0                          # tight at d=0
    return B, g, lo, hi, A, a


def test_qp_kkt_random():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(250):
        B, g, lo, hi, A, a = _rand_qp(rng, degenerate=True)
        res = solve_box_linear_qp_torch(B, g, lo, hi, A=A, a=a)
        assert res.converged
        worst = max(worst, _kkt_violation(B, g, lo, hi, A, a, res))
    assert worst < 1e-7, worst


def test_qp_matches_cvxpy():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(80):
        B, g, lo, hi, A, a = _rand_qp(rng)
        d = solve_box_linear_qp_torch(B, g, lo, hi, A=A, a=a).d.numpy()
        P = g.size
        x = cp.Variable(P)
        cons = [x >= lo, x <= hi] + ([A @ x >= a] if A is not None else [])
        cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(B)) + g @ x), cons).solve(solver=cp.CLARABEL)
        obj = lambda z: 0.5 * z @ B @ z + g @ z  # noqa: E731
        worst = max(worst, abs(obj(d) - float(0.5 * x.value @ B @ x.value + g @ x.value)))
    assert worst < 1e-6, worst


def test_qp_one_dimensional_clip():
    # unconstrained min -g/B = -1.5, clipped to the box [-1, 1]
    res = solve_box_linear_qp_torch(np.array([[2.0]]), np.array([3.0]), np.array([-1.0]), np.array([1.0]))
    assert res.converged
    assert abs(res.d.item() - (-1.0)) < 1e-10


def test_qp_no_linear_constraints():
    B = np.diag([1.0, 4.0])
    g = np.array([0.5, -2.0])
    res = solve_box_linear_qp_torch(B, g, -np.ones(2), np.ones(2))
    # min is (-g/B) = (-0.5, 0.5), both inside the box
    assert np.allclose(res.d.numpy(), np.array([-0.5, 0.5]), atol=1e-10)


def test_qp_degenerate_redundant_tiny():
    B = np.diag([1.0, 2.0])
    g = np.array([1.0, -1.0])
    lo, hi = -np.ones(2), np.ones(2)
    A = np.array([[1e-4, 0.0], [1e-4, 0.0], [0.0, 1.0]])  # rows 0,1 redundant + tiny -> d0>=0
    a = np.array([0.0, 0.0, -0.5])
    res = solve_box_linear_qp_torch(B, g, lo, hi, A=A, a=a)
    assert res.converged
    C, b = _assemble(2, lo, hi, A, a)
    assert float((C @ res.d.numpy() - b).min()) > -1e-9
    assert _kkt_violation(B, g, lo, hi, A, a, res) < 1e-7


@pytest.mark.native
def test_qp_native_parity():
    if not purcsolver.native_available():
        pytest.skip("native core not built")
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(400):
        B, g, lo, hi, A, a = _rand_qp(rng, degenerate=True)
        dref = solve_box_linear_qp_torch(B, g, lo, hi, A=A, a=a).d.numpy()
        dnat = solve_box_linear_qp(B, g, lo, hi, A=A, a=a, prefer_native=True).d.numpy()
        worst = max(worst, float(np.abs(dnat - dref).max()))
    assert worst < 1e-8, worst
