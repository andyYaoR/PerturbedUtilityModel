"""
Data-generating process for the static PURC model (random-walk closure).

Given a true parameter ``theta_0``, the model's predicted flow ``x*_b(theta_0)``
on an OD pair ``b`` defines a Markov chain on the network: starting at the
origin, the traveler picks the next link with probability proportional to ``x*``
over the node's *outgoing* links, absorbing at the destination.  Each trip is a
binary link-incidence vector ``y`` with ``E[y | b] = x*_b(theta_0)``; over ``D_b``
i.i.d. trips the link counts are ``n ~ Binomial(D_b, x*)``.

Link orientation (tail vs head) is read from the sign of the incidence matrix
``A`` (``+1`` at the tail, ``-1`` at the head, matching the demand convention
``b[origin] = +1, b[dest] = -1``).  Negative link utilities make ``x*`` acyclic,
so the walk terminates; a step cap guards degenerate cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..estimators.base import SimulatedData
from .constraints.general import GeneralPolytope
from .problem import PUMProblem
from .utils.torch_compat import DEFAULT_DTYPE, as_tensor, to_numpy


@dataclass
class ODSpec:
    """
    One origin--destination demand with a trip count.

    Attributes:
        b: Demand vector, shape ``[k]`` (``+1`` at origin, ``-1`` at destination).
        origin: Origin node index.
        dest: Destination node index.
        D: Number of i.i.d. trips ``D_b`` to sample.

    """

    b: np.ndarray
    origin: int
    dest: int
    D: int


class RandomWalkSampler:
    """
    Sample trips from a PURC predicted flow by the random-walk closure.

    Args:
        constraint: The (incidence) polytope; supplies ``A`` and its orientation.
        rng: A numpy random generator.

    Raises:
        ValueError: If the constraint is not a node-arc incidence matrix.

    """

    def __init__(self, constraint: GeneralPolytope, rng: np.random.Generator) -> None:
        if not constraint.is_incidence:
            raise ValueError(
                "RandomWalkSampler requires an incidence (node-arc) constraint; "
                "the random-walk closure is defined on a network."
            )
        self.rng = rng
        A = constraint.A.tocsc()
        n_nodes, n_edges = A.shape
        self.n_nodes = int(n_nodes)
        self.n_edges = int(n_edges)
        indptr, indices, data = A.indptr, A.indices, A.data
        tail = np.empty(n_edges, dtype=np.int64)
        head = np.empty(n_edges, dtype=np.int64)
        for e in range(n_edges):
            seg = slice(indptr[e], indptr[e + 1])
            rows, vals = indices[seg], data[seg]
            tail[e] = int(rows[vals > 0][0])  # +1 endpoint
            head[e] = int(rows[vals < 0][0])  # -1 endpoint
        self.tail, self.head = tail, head
        # Outgoing edges per node (edges whose tail is that node).
        self._out: List[np.ndarray] = [np.where(tail == u)[0] for u in range(n_nodes)]

    def sample_trip(
        self, xstar: np.ndarray, origin: int, dest: int, max_steps: int
    ) -> np.ndarray:
        """
        Sample one origin->destination trip; return its binary link-incidence.

        Args:
            xstar: Predicted link flow ``x*`` for this OD pair, shape ``[N]``.
            origin: Origin node.
            dest: Destination node.
            max_steps: Maximum links before giving up (degenerate guard).

        Returns:
            A ``[N]`` array of ``0/1`` marking the links the trip traversed.

        """
        y = np.zeros(self.n_edges, dtype=np.float64)
        x = np.clip(np.asarray(xstar, dtype=np.float64), 0.0, None)
        u = origin
        for _ in range(max_steps):
            if u == dest:
                break
            out = self._out[u]
            if out.size == 0:
                break
            p = x[out]
            s = p.sum()
            if s <= 0.0:
                break
            e = int(self.rng.choice(out, p=p / s))
            y[e] = 1.0
            u = int(self.head[e])
        return y

    def sample_od(
        self, xstar: np.ndarray, od: ODSpec, max_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample ``D`` trips for one OD pair and return link counts and frequencies.

        Args:
            xstar: Predicted link flow ``x*`` for this OD pair, shape ``[N]``.
            od: The OD specification (origin, destination, trip count).
            max_steps: Maximum links per trip.

        Returns:
            ``(n_counts, ybar)`` each ``[N]``: integer counts and ``counts / D``.

        """
        # All D trips advance in lockstep: at each step the active trips are grouped
        # by current node and their next links are drawn together (a vectorized
        # categorical per node).  This preserves the exact per-trip path law -- and
        # hence the within-OD cross-link dependence the variance estimator needs --
        # at a fraction of the cost of D sequential Python walks.
        x = np.clip(np.asarray(xstar, dtype=np.float64), 0.0, None)
        n_counts = np.zeros(self.n_edges, dtype=np.float64)
        pos = np.full(od.D, od.origin, dtype=np.int64)
        for _ in range(max_steps):
            active = np.where(pos != od.dest)[0]
            if active.size == 0:
                break
            nodes = pos[active]
            for u in np.unique(nodes):
                out = self._out[int(u)]
                here = active[nodes == u]
                if out.size == 0:
                    continue  # dead end: these trips stop where they are
                p = x[out]
                s = p.sum()
                if s <= 0.0:
                    continue
                choices = self.rng.choice(out, size=here.size, p=p / s)
                np.add.at(n_counts, choices, 1.0)
                pos[here] = self.head[choices]
        return n_counts, n_counts / od.D


def simulate_dataset(
    problem: PUMProblem,
    solver,
    theta0,
    od_specs: Sequence[ODSpec],
    rng: np.random.Generator,
    *,
    max_steps: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> SimulatedData:
    """
    Simulate per-OD link data from the PURC model at ``theta_0``.

    Solves ``x*_b(theta_0)`` for all OD pairs in one batched forward solve, then
    samples ``D_b`` random-walk trips per OD pair.

    Args:
        problem: The PURC forward problem (incidence constraint).
        solver: A preprocessed forward solver exposing ``solve_batch``.
        theta0: True parameter ``(beta_0, gamma_0)``.
        od_specs: The OD specifications (demand, origin, destination, trips).
        rng: A numpy random generator.
        max_steps: Maximum links per trip (defaults to ``4 * n_nodes``).
        meta: Optional provenance recorded on the result.

    Returns:
        A :class:`~purc.estimators.base.SimulatedData` bundle.

    """
    c = problem.constraint
    sampler = RandomWalkSampler(c, rng)
    if max_steps is None:
        max_steps = 4 * c.num_constraints
    b_batch = np.stack([np.asarray(od.b, dtype=np.float64).ravel() for od in od_specs])
    res = solver.solve_batch(theta0, b_batch)
    xstar0 = to_numpy(res.x)  # [B, N]

    n_counts = np.zeros_like(xstar0)
    ybar = np.zeros_like(xstar0)
    for i, od in enumerate(od_specs):
        n_counts[i], ybar[i] = sampler.sample_od(xstar0[i], od, max_steps)
    D = np.array([od.D for od in od_specs], dtype=np.float64)
    # Bridge the (numpy-sampled) data back to torch -- the package is torch-native.
    return SimulatedData(
        n_counts=as_tensor(n_counts).to(DEFAULT_DTYPE),
        ybar=as_tensor(ybar).to(DEFAULT_DTYPE),
        D=as_tensor(D).to(DEFAULT_DTYPE),
        b_batch=as_tensor(b_batch).to(DEFAULT_DTYPE),
        xstar0=as_tensor(xstar0).to(DEFAULT_DTYPE),
        meta=meta or {},
    )
