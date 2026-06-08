"""
Shared builders and the theta_0 catalog for the estimation Monte Carlo study.

Provides synthetic and TNTP network problems with a link-attribute matrix ``Z``
chosen so that all utilities ``v = Z beta_0`` are negative (acyclic flows, the
DGP precondition), OD-pair generators, and the catalog of true parameters used
across the claim experiments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from purc.static_purc import PUMProblem
from purc.static_purc.constraints import GeneralPolytope
from purc.static_purc.dgp import ODSpec
from purc.static_purc.perturbations import get_perturbation

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data") + os.sep


@dataclass(frozen=True)
class Theta0:
    """A named true parameter: utility coefficients ``beta`` and sieve ``gamma``."""

    name: str
    beta: np.ndarray
    gamma: np.ndarray

    @property
    def L(self) -> int:
        """Highest sieve degree ``L`` (``= len(gamma) + 2``)."""
        return int(self.gamma.size) + 2


# Catalog of true parameters (gamma chosen feasible; the "neg" case is in Gamma_B
# but outside R_+, so only the Bernstein projection can recover it).
CATALOG = {
    "pos3": Theta0("pos3", beta=np.array([-1.0, -1.0]), gamma=np.array([0.5])),
    "pos345": Theta0("pos345", beta=np.array([-1.0, -1.0]), gamma=np.array([0.5, 0.3, 0.1])),
    "neg": Theta0("neg", beta=np.array([-1.0, -1.0]), gamma=np.array([0.8, -0.3])),
}


def synth_incidence(n_nodes: int, seed: int) -> np.ndarray:
    """
    Random connected directed graph incidence matrix ``[n_nodes, N]``.

    The graph is *simple*: at most one arc per unordered node pair (no parallel
    and no anti-parallel "two-way" arcs), so the weighted-Laplacian normal
    equations route to the fast ``PURCLaplacianSolver`` incidence path rather than
    the general SDDM backend.
    """
    rng = np.random.default_rng(seed)
    used = set()  # unordered node pairs already connected by an arc
    edges = []
    for i in range(n_nodes):  # a directed cycle guarantees connectivity
        a, b = i, (i + 1) % n_nodes
        edges.append((a, b))
        used.add(frozenset((a, b)))
    for _ in range(4 * n_nodes):
        a, b = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
        if a != b and frozenset((a, b)) not in used:
            edges.append((a, b))
            used.add(frozenset((a, b)))
    inc = np.zeros((n_nodes, len(edges)))
    for k, (a, b) in enumerate(edges):
        inc[a, k], inc[b, k] = 1.0, -1.0
    return inc


def _attribute_matrix(N: int, K: int, seed: int) -> sp.csr_matrix:
    """Positive link-attribute matrix ``Z`` (so ``Z beta`` is negative for beta<0)."""
    rng = np.random.default_rng(seed + 777)
    cols = [np.ones(N)]  # an intercept-like attribute
    for _ in range(K - 1):
        cols.append(rng.uniform(0.5, 1.5, N))
    return sp.csr_matrix(np.stack(cols, axis=1))


def _attribute_matrix_real(N: int, K: int, attrs: np.ndarray) -> sp.csr_matrix:
    """
    ``Z`` from a real positive link attribute (e.g. free-flow time).

    Column 0 is an intercept; the remaining ``K-1`` columns are the attribute
    rescaled to mean one (and its powers if ``K>2``), so ``Z`` stays positive and
    ``Z beta`` is negative for ``beta<0`` (the acyclic-flow DGP precondition).
    """
    a = np.asarray(attrs, float)
    a = a / max(float(a.mean()), 1e-9)
    a = np.clip(a, 0.1, None)
    cols = [np.ones(N)] + [a ** (j + 1) for j in range(K - 1)]
    return sp.csr_matrix(np.stack(cols, axis=1))


def build_network(network: str, seed: int) -> Tuple[np.ndarray, "np.ndarray | None"]:
    """
    Return ``(incidence[n,N], attrs[N] | None)`` for a named network.

    ``network`` is either ``"synth-<n>"`` (random simple graph, no real attribute)
    or a TNTP network name (``"SiouxFalls"``, ``"ChicagoSketch"``,
    ``"ChicagoRegional"``) whose free-flow times serve as the link attribute.  Real
    networks have two-way streets (anti-parallel arcs), so they route to the general
    SDDM path rather than the incidence fast path.
    """
    if network.startswith("synth"):
        n = int(network.split("-")[1]) if "-" in network else 30
        return synth_incidence(n, seed), None
    inc, fftt = tntp_incidence(f"{DATA}{network}_net.tntp")
    return inc, fftt


def make_problem(
    inc: np.ndarray, th0: Theta0, seed: int, attrs: "np.ndarray | None" = None
) -> PUMProblem:
    """
    Build a PURC problem on ``inc`` with attributes ``Z`` and the sieve at ``th0``.

    If ``attrs`` is given (a real per-link attribute, e.g. TNTP free-flow time) the
    utility design ``Z`` is built from it; otherwise ``Z`` is synthetic random.
    """
    n, N = inc.shape
    K = int(th0.beta.size)
    Z = _attribute_matrix(N, K, seed) if attrs is None else _attribute_matrix_real(N, K, attrs)
    pert = get_perturbation("polynomial_sieve", gamma=th0.gamma)
    d0 = np.zeros(n)
    d0[0], d0[1] = 1.0, -1.0
    cons = GeneralPolytope(sp.csr_matrix(inc), d0, ell=np.ones(N))
    prob = PUMProblem(pert, cons, Z=Z)
    # DGP precondition: all utilities strictly negative.
    v = Z @ th0.beta
    assert np.all(v < 0), "utilities must be negative for acyclic flows"
    return prob


def make_ods(n_nodes: int, B: int, D: int, seed: int) -> List[ODSpec]:
    """Generate ``B`` random OD pairs (distinct origin/destination), each with ``D`` trips."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        o, t = rng.choice(n_nodes, 2, replace=False)
        d = np.zeros(n_nodes)
        d[o], d[t] = 1.0, -1.0
        out.append(ODSpec(b=d, origin=int(o), dest=int(t), D=D))
    return out


def tntp_incidence(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a TNTP network: return (incidence dense ``[n,N]``, free-flow times)."""
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from tntp import load_net

    net = load_net(path)
    return net.A.toarray(), np.asarray(net.fftt, dtype=float)
