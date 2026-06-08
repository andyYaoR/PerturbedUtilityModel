"""
Estimate a perturbed-utility model on a real road network (SiouxFalls).

This example applies the same debiased Fenchel-Young workflow as
``examples/estimate.py`` to the SiouxFalls network, using a two-dimensional utility
in which each link carries a free-flow travel time and a synthetic congestion-related
attribute (its inverse capacity), so that ``v = Z @ beta`` with
``beta = (beta_time, beta_congestion)``. The example simulates route-choice trips at
a known parameter, recovers both utility coefficients and the cubic sieve
coefficient, and reports sandwich standard errors.

Inverse capacity is used as the second attribute, rather than link length, because
on SiouxFalls the length and free-flow-time columns are identical: their
coefficients would not be separately identified and the sandwich information matrix
would be singular. Capacity is the one link attribute uncorrelated with travel time,
so its inverse supplies a second identified direction, and it reads naturally as a
disutility (travelers avoid low-capacity, congestion-prone links).

The data-generating process requires strictly negative link utilities, so that the
predicted flows are acyclic and the random-walk sampler terminates. Both attribute
columns are strictly positive and both coefficients are negative, ensuring ``v < 0``.
Each attribute is kept in a fixed, interpretable unit -- free-flow time in its native
minutes, and capacity measured per ``10^4`` veh/h so the inverse-capacity attribute is
of order one. This is a deliberate unit choice (a fixed scale constant), not a
data-dependent rescaling by the sample mean; in raw veh/h units the same model would
carry a congestion coefficient of order ``10^4``, since ``1 / capacity`` is ``~1e-4``.

Run:  python examples/estimate_siouxfalls.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

from purc.estimators.debiased_fy import (
    DebiasedFYEstimator,
    DebiasedFYLoss,
    EstimatorConfig,
    GammaProjection,
    sandwich_variance,
)
from purc.static_purc import ForwardSolverConfig, PUMProblem
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.dgp import ODSpec, simulate_dataset
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.utils.torch_compat import to_numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tntp import load_net  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "SiouxFalls_net.tntp")

# Fixed reference capacity (veh/h) used to put the inverse-capacity attribute on an
# O(1) scale: the congestion attribute is (reference capacity) / (link capacity).
_CAP_UNIT = 1.0e4


def link_attributes(net) -> np.ndarray:
    """
    Build a positive ``(N, 2)`` attribute matrix: free-flow time and inverse capacity.

    Both attributes are kept in fixed, interpretable units so the coefficients are
    O(1): free-flow time in its native minutes, and capacity measured per
    :data:`_CAP_UNIT` veh/h so the congestion attribute ``_CAP_UNIT / capacity`` is of
    order one. This is a fixed unit choice, not a data-dependent rescaling by the
    sample mean. Nonpositive entries (e.g. centroid connectors) take the column mean.

    Args:
        net: A parsed :class:`tntp.TNTPNetwork`.

    Returns:
        A strictly-positive ``(N, 2)`` attribute matrix ``[free-flow time, _CAP_UNIT / capacity]``.

    """
    cap = np.asarray(net.capacity, dtype=float)
    congestion = np.divide(_CAP_UNIT, cap, out=np.zeros_like(cap), where=cap > 0)
    cols = []
    for raw in (net.fftt, congestion):
        x = np.asarray(raw, dtype=float)
        pos = x[x > 0]
        fill = float(pos.mean()) if pos.size else 1.0
        x = np.where(x > 0, x, fill)  # replace nonpositive entries with the column mean
        cols.append(x)
    return np.column_stack(cols)


def make_ods(n_nodes: int, n_od: int, D: int, seed: int) -> list[ODSpec]:
    """Draw ``n_od`` distinct OD pairs, each with ``D`` trips."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_od):
        o, t = rng.choice(n_nodes, 2, replace=False)
        d = np.zeros(n_nodes)
        d[o], d[t] = 1.0, -1.0
        out.append(ODSpec(b=d, origin=int(o), dest=int(t), D=D))
    return out


def main() -> None:
    """Estimate (beta_time, beta_congestion, gamma_3) on SiouxFalls from simulated trips."""
    net = load_net(_DATA)
    A, n, N = net.A, net.n_nodes, net.n_links

    # --- truth ---------------------------------------------------------------
    beta0 = np.array([-0.5, -1.0])  # (free-flow time, congestion) coefficients (negative)
    gamma0 = np.array([0.5])  # one cubic sieve coefficient -> L = 3
    L = gamma0.size + 2

    Z = link_attributes(net)  # (N, 2), strictly positive
    assert np.all(Z @ beta0 < 0), "DGP needs strictly negative utilities v = Z @ beta0"

    d0 = np.zeros(n)
    d0[0], d0[1] = 1.0, -1.0
    cons = GeneralPolytope(A, d0, ell=np.ones(N))
    prob = PUMProblem(get_perturbation("polynomial_sieve", gamma=gamma0), cons, Z=sp.csr_matrix(Z))

    # ONE preprocessed solver, reused for simulate + fit + inference.
    solver = IPMSolver(ForwardSolverConfig(max_iter=200), crossover=False, safeguard=True)
    solver.preprocess(prob)

    B, D = 40, 3000
    ods = make_ods(n, B, D, seed=7)
    print(f"SiouxFalls: {n} nodes, {N} links | Data: B={B} OD pairs x D={D} trips")
    print(f"True parameter: beta_time={beta0[0]:+.2f}  beta_congestion={beta0[1]:+.2f}  "
          f"gamma_3={gamma0[0]:+.2f}\n")
    data = simulate_dataset(prob, solver, (beta0, gamma0), ods, np.random.default_rng(8))

    # --- estimate ------------------------------------------------------------
    loss = DebiasedFYLoss(prob, solver, data, L=L)
    est = DebiasedFYEstimator(
        prob, solver, L=L,
        config=EstimatorConfig(proj=GammaProjection("bernstein")),
        theta_init=loss.layout.pack(np.array([-0.2, -0.2]), np.array([0.0])),
    )
    res = est.fit(data)
    print(f"Fit: converged={res.converged}  outer_iters={res.n_outer}  "
          f"objective={res.objective:.6e}\n")

    # --- recovery + sandwich inference --------------------------------------
    sw = sandwich_variance(DebiasedFYLoss(prob, solver, data, L=L), res.theta_hat)
    theta_hat = to_numpy(res.theta_hat)
    theta0 = np.concatenate([beta0, gamma0])
    se = to_numpy(sw.se)
    names = ["beta_time", "beta_congest", "gamma_3"]
    print(f"Recovery and sandwich inference (cond(bread)={sw.cond_A:.1e}):")
    print(f"  {'param':12s} {'true':>7s} {'estimate':>9s} {'SE':>8s} "
          f"{'z(true)':>8s} {'95% CI':>8s}")
    for j, name in enumerate(names):
        z = (theta_hat[j] - theta0[j]) / se[j]
        covered = "yes" if abs(theta_hat[j] - theta0[j]) <= 1.96 * se[j] else "no"
        print(f"  {name:12s} {theta0[j]:>7.2f} {theta_hat[j]:>9.3f} {se[j]:>8.3f} "
              f"{z:>8.2f} {covered:>8s}")
    print("\nEstimation completed on SiouxFalls (2 utility coefficients and 1 sieve shape).")


if __name__ == "__main__":
    main()
