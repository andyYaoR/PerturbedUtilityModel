"""
Prototype log-barrier continuation for hard Chicago PURC instances.

This is a development probe, not the production solver.  It solves fixed-mu
smooth dual barrier subproblems with Armijo globalization, then hands the final
multiplier to the exact box SSN solver for polishing at mu=0.

Run:
    python dev/prototype_barrier_chicago.py regional --perturbation sieve --beta 5
    python dev/prototype_barrier_chicago.py sketch --perturbation quadratic --beta 5
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from purc.static_purc import PUMProblem, SSNConfig
from purc.static_purc.backends.routing import LaplacianBackend
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.perturbations import get_perturbation
from purc.static_purc.solvers.barrier import (
    BarrierRecoveryConfig,
    barrier_dual_objective,
    recover_barrier_primal,
)
from purc.static_purc.solvers import RegularizedSSNSolver
from purc.static_purc.utils.torch_compat import as_tensor
from tntp import load_net

DATA = "/Users/ruiyao/Library/CloudStorage/Dropbox/Technion/Codes/LaplacianSolve/examples/data/"
NETS = {
    "sketch": DATA + "ChicagoSketch_net.tntp",
    "regional": DATA + "ChicagoRegional_net.tntp",
}


@dataclass
class BarrierConfig:
    """Configuration for the development barrier continuation probe."""

    tol: float = 1e-9
    max_outer: int = 8
    max_inner: int = 80
    mu_factor: float = 0.2
    armijo_c1: float = 1e-4
    armijo_beta: float = 0.5
    max_linesearch: int = 40
    eps_floor: float = 1e-12
    root_max_iter: int = 80
    root_xtol: float = 1e-13
    polish_iter: int = 200


def feasible_ods(n_nodes: int, n_od: int, seed: int = 1) -> list[tuple[int, int]]:
    """Return deterministic OD pairs matching the benchmark generator."""
    rng = np.random.default_rng(seed)
    ods = []
    while len(ods) < n_od:
        o, d = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
        if o != d:
            ods.append((o, d))
    return ods


def perturbation_spec(name: str):
    """Return ``(display_name, perturbation, gamma)`` for a diagnostic kernel."""
    if name == "quadratic":
        return "quadratic", get_perturbation("quadratic"), np.zeros(0)
    if name == "entropy":
        return "entropy", get_perturbation("entropy"), np.zeros(0)
    if name == "sieve":
        gamma = np.array([0.3, 0.1])
        return "sieve(L=4)", get_perturbation("polynomial_sieve", gamma=gamma), gamma
    raise ValueError(f"unknown perturbation {name!r}")


def barrier_recovery(
    problem: PUMProblem,
    v: torch.Tensor,
    lam: torch.Tensor,
    gamma,
    mu: float,
    cfg: BarrierConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover interior ``x(lambda, mu)`` and Newton weights for the log barrier."""
    recovery_cfg = BarrierRecoveryConfig(
        root_max_iter=cfg.root_max_iter,
        root_xtol=cfg.root_xtol,
    )
    return recover_barrier_primal(problem, v, lam, gamma, mu, recovery_cfg)


def barrier_phi(
    problem: PUMProblem,
    v: torch.Tensor,
    lam: torch.Tensor,
    b_rhs: torch.Tensor,
    gamma,
    mu: float,
    x: torch.Tensor,
) -> float:
    """Evaluate the convex dual objective for the fixed-mu barrier subproblem."""
    return barrier_dual_objective(problem, v, lam, b_rhs, gamma, mu, x=x)


def solve_barrier_path(
    problem: PUMProblem,
    v_np: np.ndarray,
    gamma_np: np.ndarray,
    b_np: np.ndarray,
    cfg: BarrierConfig,
) -> dict:
    """Solve the barrier path and return diagnostics plus a polished SSN result."""
    backend = LaplacianBackend(problem.constraint, {})
    v = as_tensor(v_np).reshape(-1)
    b_rhs = as_tensor(b_np).reshape(-1)
    gamma = as_tensor(gamma_np).reshape(-1)
    lam = torch.zeros(problem.constraint.num_constraints, dtype=torch.float64)
    scale = float(torch.median(torch.abs(v) + problem.constraint.ell)) + 1.0
    mu = scale
    stages = []
    total_inner = 0

    with torch.no_grad():
        for outer in range(cfg.max_outer):
            stage_tol = max(cfg.tol, min(1e-3, 0.05 * mu / scale))
            accepted_steps = 0
            last_alpha = 1.0
            for inner in range(cfg.max_inner):
                total_inner += 1
                x, weight = barrier_recovery(problem, v, lam, gamma, mu, cfg)
                r = problem.constraint.matvec(x) - b_rhs
                r_inf = float(r.abs().max()) if r.numel() else 0.0
                if r_inf < stage_tol:
                    break

                phi0 = barrier_phi(problem, v, lam, b_rhs, gamma, mu, x)
                direction = backend.solve(weight, cfg.eps_floor, -r)
                rd = float(r @ direction)
                if (not math.isfinite(rd)) or rd >= 0:
                    raise RuntimeError(f"non-descent barrier direction at outer={outer}: rd={rd}")

                alpha = 1.0
                accepted = False
                for _ in range(cfg.max_linesearch):
                    lam_trial = lam + alpha * direction
                    x_trial, _ = barrier_recovery(problem, v, lam_trial, gamma, mu, cfg)
                    phi_trial = barrier_phi(problem, v, lam_trial, b_rhs, gamma, mu, x_trial)
                    if math.isfinite(phi_trial) and phi_trial <= phi0 + cfg.armijo_c1 * alpha * rd:
                        lam = lam_trial
                        accepted = True
                        accepted_steps += 1
                        last_alpha = alpha
                        break
                    alpha *= cfg.armijo_beta
                if not accepted:
                    raise RuntimeError(
                        f"barrier line search failed at outer={outer}, inner={inner}, "
                        f"residual={r_inf:.3e}, mu={mu:.3e}"
                    )

            x, _ = barrier_recovery(problem, v, lam, gamma, mu, cfg)
            r = problem.constraint.matvec(x) - b_rhs
            r_inf = float(r.abs().max()) if r.numel() else 0.0
            stages.append(
                {
                    "outer": outer,
                    "mu": mu,
                    "tol": stage_tol,
                    "residual": r_inf,
                    "inner": inner + 1,
                    "accepted": accepted_steps,
                    "alpha": last_alpha,
                }
            )
            if r_inf < cfg.tol:
                break
            mu *= cfg.mu_factor

    polish = RegularizedSSNSolver(
        SSNConfig(tol=cfg.tol, max_iter=cfg.polish_iter, warm_start=False)
    )
    polish.preprocess(problem)
    res = polish.solve((v_np, gamma_np), b=b_np, lam0=lam)
    return {"stages": stages, "total_inner": total_inner, "polish": res}


def run(args: argparse.Namespace) -> None:
    """Run the barrier prototype on selected Chicago OD pairs."""
    net = load_net(NETS[args.network])
    cost = np.maximum(net.length, 1e-3)
    title, perturbation, gamma = perturbation_spec(args.perturbation)
    v = -args.beta * cost
    ods = feasible_ods(net.n_nodes, args.n_od, seed=args.seed)
    poly = GeneralPolytope(net.A, np.zeros(net.n_nodes), ell=cost, validate=False)
    prob = PUMProblem(perturbation, poly)
    cfg = BarrierConfig(
        tol=args.tol,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        mu_factor=args.mu_factor,
        polish_iter=args.polish_iter,
    )

    print(
        f"Barrier prototype: Chicago {args.network}, n={net.n_nodes}, m={net.n_links}, "
        f"perturbation={title}, beta={args.beta:g}, tol={args.tol:g}"
    )
    print(f"OD pairs: {ods}")

    for idx, (origin, destination) in enumerate(ods):
        b = np.zeros(net.n_nodes)
        b[origin] = 1.0
        b[destination] = -1.0
        t0 = time.perf_counter()
        out = solve_barrier_path(prob, v, gamma, b, cfg)
        elapsed = (time.perf_counter() - t0) * 1e3
        res = out["polish"]
        print(
            f"\nOD {idx}: {origin}->{destination} success={res.success} "
            f"polish_nit={res.nit} residual={res.residual:.3e} "
            f"barrier_inner={out['total_inner']} elapsed_ms={elapsed:.1f}"
        )
        for stage in out["stages"]:
            print(
                "  "
                f"mu={stage['mu']:.3e} tol={stage['tol']:.1e} "
                f"resid={stage['residual']:.3e} inner={stage['inner']:3d} "
                f"steps={stage['accepted']:3d} alpha={stage['alpha']:.1e}"
            )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", choices=sorted(NETS), nargs="?", default="regional")
    parser.add_argument(
        "--perturbation",
        choices=["quadratic", "entropy", "sieve"],
        default="sieve",
    )
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--n-od", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-outer", type=int, default=8)
    parser.add_argument("--max-inner", type=int, default=80)
    parser.add_argument("--mu-factor", type=float, default=0.2)
    parser.add_argument("--polish-iter", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
