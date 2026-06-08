"""
Estimate a perturbed-utility route-choice model from observed trip data.

This example demonstrates the complete estimation workflow for the debiased
Fenchel-Young estimator, which recovers both the utility coefficients ``beta`` and
the perturbation shape ``gamma`` from observed route-choice frequencies. 

The workflow has five steps:

  1. Choose a true parameter: utility coefficients ``beta_0`` and a polynomial-sieve
     perturbation shape ``gamma_0`` (here a single cubic coefficient ``gamma_0_3``).
  2. Simulate data: solve the model's predicted link flows ``x*(theta_0)`` and draw
     ``D`` random-walk trips per origin-destination pair, producing the observed
     link-traversal frequencies.
  3. Estimate ``(beta, gamma)`` from those frequencies with
     :class:`~purc.estimators.debiased_fy.DebiasedFYEstimator` and compare with the
     true parameter.
  4. Inference: sandwich standard errors, z-statistics, and 95%
     confidence intervals.
  5. Illustrate the effect of debiasing: at the true parameter the debiased score is
     approximately zero, whereas the naive plug-in score is biased on the sieve
     coefficients.

Run:  python examples/estimate.py
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from purc.estimators.debiased_fy import (
    DebiasedFYEstimator,
    DebiasedFYLoss,
    EstimatorConfig,
    GammaProjection,
    NaiveFYLoss,
    sandwich_variance,
)
from purc.static_purc import ForwardSolverConfig, PUMProblem
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.dgp import ODSpec, simulate_dataset
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers.ipm import IPMSolver
from purc.static_purc.utils.torch_compat import to_numpy


def build_incidence(n_nodes: int, seed: int) -> np.ndarray:
    """
    Random connected directed graph incidence ``[n_nodes, N]``.

    A directed cycle guarantees strong connectivity; extra random arcs add
    alternative routes.

    Args:
        n_nodes: Number of nodes.
        seed: RNG seed.

    Returns:
        Dense node-arc incidence (``+1`` tail, ``-1`` head).

    """
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    for _ in range(4 * n_nodes):
        a, b = rng.integers(0, n_nodes, 2)
        if a != b:
            edges.append((int(a), int(b)))
    inc = np.zeros((n_nodes, len(edges)))
    for k, (a, b) in enumerate(edges):
        inc[a, k], inc[b, k] = 1.0, -1.0
    return inc


def build_problem(inc: np.ndarray, gamma0: np.ndarray, beta0: np.ndarray, seed: int) -> PUMProblem:
    """
    Assemble the PURC problem: positive link attributes ``Z``, a sieve kernel, and
    a single-commodity flow polytope.

    The DGP precondition is that link utilities ``v = Z @ beta0`` are strictly
    negative (so ``x*`` is acyclic and the random walk terminates fast).

    Args:
        inc: Node-arc incidence ``[n, N]``.
        gamma0: True sieve coefficients ``[L-2]``.
        beta0: True utility coefficients ``[K]`` (negative in the route choice example).
        seed: RNG seed for the attribute matrix.

    Returns:
        The configured :class:`PUMProblem`.

    """
    n, N = inc.shape
    K = beta0.size
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0.5, 1.5, (N, K))  # strictly positive link attributes
    assert np.all(Z @ beta0 < 0), "DGP needs strictly negative utilities v = Z @ beta0"
    pert = get_perturbation("polynomial_sieve", gamma=gamma0)
    # Reference demand vector, can be override during solve.
    d0 = np.zeros(n)
    d0[0], d0[1] = 1.0, -1.0  # +1 at the origin (source), -1 at the destination (sink)
    cons = GeneralPolytope(sp.csr_matrix(inc), d0, ell=np.ones(N))
    return PUMProblem(pert, cons, Z=sp.csr_matrix(Z))


def make_ods(n_nodes: int, n_od: int, D: int, seed: int) -> list[ODSpec]:
    """
    Draw ``n_od`` distinct origin/destination pairs, each with ``D`` trips.

    Args:
        n_nodes: Number of nodes.
        n_od: Number of OD pairs.
        D: Trips per OD pair.
        seed: RNG seed.

    Returns:
        A list of :class:`ODSpec`.

    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_od):
        o, t = rng.choice(n_nodes, 2, replace=False)
        d = np.zeros(n_nodes)
        d[o], d[t] = 1.0, -1.0
        out.append(ODSpec(b=d, origin=int(o), dest=int(t), D=D))
    return out


def main() -> None:
    """Run the debiased Fenchel--Young estimation example end to end."""
    # --- 1. truth ------------------------------------------------------------
    beta0 = np.array([-2.0])  # one utility coefficient (negative)
    gamma0 = np.array([0.5])  # one cubic sieve coefficient -> L = len(gamma)+2 = 3
    L = gamma0.size + 2

    # --- 2. problem + simulated data ----------------------------------------
    inc = build_incidence(n_nodes=8, seed=12)
    prob = build_problem(inc, gamma0, beta0, seed=13)
    n = inc.shape[0]
    # Prepare solver, reused for simulate + fit + inference.
    solver = IPMSolver(ForwardSolverConfig(max_iter=200), crossover=False, safeguard=True)
    solver.preprocess(prob)

    B, D = 60, 3000
    ods = make_ods(n, B, D, seed=14)
    print(f"Network: {n} nodes, {inc.shape[1]} links | Data: B={B} OD pairs x D={D} trips")
    print(f"True parameter: beta_0={beta0}  gamma_0_3={gamma0}\n")
    data = simulate_dataset(prob, solver, (beta0, gamma0), ods, np.random.default_rng(15))

    # --- 3. estimate ---------------------------------------------------------
    loss = DebiasedFYLoss(prob, solver, data, L=L)
    est = DebiasedFYEstimator(
        prob, solver, L=L,
        config=EstimatorConfig(proj=GammaProjection("bernstein")),
        theta_init=loss.layout.pack(np.array([-1.0]), np.array([0.0])),  # start away from truth
    )
    res = est.fit(data)
    beta_hat = to_numpy(res.beta_hat)
    gamma_hat = to_numpy(res.gamma_hat)
    print(f"Fit: converged={res.converged}  outer_iters={res.n_outer}  "
          f"objective={res.objective:.6e}")
    print(f"  beta:    true={beta0[0]:+.3f}  est={beta_hat[0]:+.3f}  "
          f"abs.err={abs(beta_hat[0] - beta0[0]):.3f}")
    print(f"  gamma_0_3: true={gamma0[0]:+.3f}  est={gamma_hat[0]:+.3f}  "
          f"abs.err={abs(gamma_hat[0] - gamma0[0]):.3f}\n")

    # --- 4. inference: sandwich standard errors ------------------------------
    # res.se is not populated by fit(); compute it from the sandwich at theta_hat.
    sw = sandwich_variance(DebiasedFYLoss(prob, solver, data, L=L), res.theta_hat)
    theta_hat = to_numpy(res.theta_hat)
    theta0 = np.concatenate([beta0, gamma0])
    se = to_numpy(sw.se)
    names = ["beta_1", "gamma_0_3"]
    print(f"Sandwich inference (cond(bread)={sw.cond_A:.1e}):")
    print(f"  {'param':8s} {'estimate':>9s} {'SE':>8s} {'z(true)':>8s} {'95% CI covers true':>20s}")
    for j, name in enumerate(names):
        z = (theta_hat[j] - theta0[j]) / se[j]
        covered = abs(theta_hat[j] - theta0[j]) <= 1.96 * se[j]
        print(f"  {name:8s} {theta_hat[j]:>9.3f} {se[j]:>8.3f} {z:>8.2f} "
              f"{('yes' if covered else 'no'):>20s}")

    # --- 5. effect of debiasing: the naive plug-in score is biased on the sieve block
    # The score should vanish at the truth.  With many OD pairs the empirical score
    # concentrates on its expectation, exposing the naive plug-in's O(1/D) bias on
    # the higher-moment (sieve / curvature) statistics; short trips (D=30) make it
    # plain.  The beta block uses only first moments and is unbiased either way.
    ods_b = make_ods(n, 300, 30, seed=24)
    data_b = simulate_dataset(prob, solver, (beta0, gamma0), ods_b, np.random.default_rng(25))
    th0 = DebiasedFYLoss(prob, solver, data_b, L=L).layout.pack(beta0, gamma0)
    _, g_deb = DebiasedFYLoss(prob, solver, data_b, L=L).value_and_grad(th0)
    _, g_naive = NaiveFYLoss(prob, solver, data_b, L=L).value_and_grad(th0)
    g_deb, g_naive = to_numpy(g_deb), to_numpy(g_naive)
    print("\nScore at the true parameter (B=300, D=30); the estimating equation sets it to zero:")
    print(f"  {'block':8s} {'debiased':>10s} {'naive':>10s}")
    print(f"  {'beta_1':8s} {abs(g_deb[0]):>10.4f} {abs(g_naive[0]):>10.4f}   "
          "first moment, unbiased either way")
    print(f"  {'gamma_0_3':8s} {abs(g_deb[1]):>10.4f} {abs(g_naive[1]):>10.4f}   "
          "naive plug-in is biased; debiasing removes it")


if __name__ == "__main__":
    main()
