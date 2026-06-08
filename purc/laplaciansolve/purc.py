r"""
PURC-specialized Laplacian solver: incidence in, batched SDDM solves out.

The PURC semismooth-Newton loop solves, every iteration, the SDDM system

.. math::

    (H^k + \varepsilon_k I)\,\Delta\lambda = -r^k, \qquad
    H^k = C_{S^k}\,\mathrm{diag}(D^k)\,C_{S^k}^\top,

a weighted graph Laplacian over the active link set plus a regularizer.  The
node-link incidence ``C`` (hence the Laplacian *sparsity pattern*) is **fixed**
for the whole run; only the edge weights ``D``, the active mask, ``eps`` and the
right-hand side change.

:class:`PURCLaplacianSolver` is constructed **once** from the edge list.  It
caches the Laplacian pattern and the edge -> CSC-position maps, and builds a
:class:`~purc.laplaciansolve.batched.BatchedSDDMSolver` on that pattern.  Each
iteration, :meth:`solve_step` / :meth:`solve_batch` take per-edge weights ``D``
(and optional active mask, ``eps``, RHS), assemble ``M = C diag(D) C^T + eps*I``
**natively** on the cached pattern (one GIL-released call, no Python loop), and
solve - returning a PUM-style result dict.

Boundary: the package is a standalone *linear* solver and embeds **no**
perturbation kernels.  The Newton solver computes the model-specific weights
(e.g. ``D_i = 1/(l_i h''(x_i; gamma))``) itself and passes finished per-edge
weights; inactive edges are deselected via ``active_mask`` (or simply ``D_i=0``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse

from ._convert import VectorAdapter, adjacency_to_csc
from ._loader import native_core
from .batched import BatchedSDDMSolver
from .reference import lap
from .solver import SolverConfig


def _incidence_to_edges(coo) -> np.ndarray:
    """
    Extract undirected edges from a node-link incidence matrix C (n x m).

    Args:
        coo: The incidence as a scipy COO matrix; each column is one edge with
            exactly two nonzeros (its endpoints).

    Returns:
        An ``(m, 2)`` int64 array of endpoint pairs.

    Raises:
        ValueError: If any column does not have exactly two nonzeros.

    """
    m = coo.shape[1]
    counts = np.bincount(coo.col, minlength=m)
    if m and not np.all(counts == 2):
        raise ValueError(
            "each incidence column must have exactly two nonzeros (one edge's endpoints)"
        )
    order = np.lexsort((coo.row, coo.col))  # group by column, rows ascending within
    return np.ascontiguousarray(coo.row[order].astype(np.int64).reshape(m, 2))


def _to_edges_and_n(graph, n_nodes):
    """
    Normalize the graph argument to an ``(m, 2)`` edge list and a node count.

    Accepts an edge list, a square adjacency / Laplacian / SDDM matrix (edges =
    off-diagonal nonzeros), or a rectangular incidence ``C`` (edges = per-column
    endpoints); the matrix forms may be scipy sparse, numpy dense, or torch (CPU).

    Args:
        graph: An ``(m, 2)`` integer edge list, or a graph matrix (see above).
        n_nodes: Node count; required for an edge list unless inferred from it,
            and validated against the matrix shape otherwise.

    Returns:
        A tuple ``(edges, n)`` of the int64 endpoint pairs and the node count.

    Raises:
        ValueError: If ``graph`` is malformed or ``n_nodes`` disagrees with it.

    """
    if not sparse.issparse(graph):
        arr = np.asarray(graph)
        if arr.ndim == 2 and arr.shape[1] == 2 and np.issubdtype(arr.dtype, np.integer):
            edges = np.ascontiguousarray(arr, dtype=np.int64)
            n = int(n_nodes) if n_nodes is not None else (int(edges.max()) + 1 if edges.size else 1)
            return edges, n
    mat = adjacency_to_csc(graph)
    mat.eliminate_zeros()
    coo = mat.tocoo()
    nr, nc = coo.shape
    if nr == nc:  # adjacency / Laplacian / SDDM: undirected off-diagonal nonzeros
        sel = coo.row != coo.col
        u = np.minimum(coo.row[sel], coo.col[sel])
        v = np.maximum(coo.row[sel], coo.col[sel])
        edges = np.unique(np.stack([u, v], axis=1).astype(np.int64), axis=0)
        n = int(nr)
    else:  # incidence C (n x m)
        edges = _incidence_to_edges(coo)
        n = int(nr)
    if n_nodes is not None and int(n_nodes) != n:
        raise ValueError(f"n_nodes={n_nodes} disagrees with matrix shape {coo.shape}")
    return np.ascontiguousarray(edges), n


def _build_pattern_and_maps(edges: np.ndarray, n: int):
    """
    Build the Laplacian CSC pattern and the edge/node -> CSC-position maps.

    Args:
        edges: ``(m, 2)`` int array of undirected edge endpoints (0-based).
        n: Number of nodes.

    Returns:
        A tuple ``(m_pattern, diag_pos, off0, off1, ediag0, ediag1)`` where
        ``m_pattern`` is a representative SDDM matrix (for routing) and the rest
        are int64 CSC-position maps consumed by the native assembler.

    Raises:
        ValueError: On self-loops, out-of-range endpoints, or duplicate edges.

    """
    m = edges.shape[0]
    u = edges[:, 0]
    v = edges[:, 1]
    if np.any(u == v):
        raise ValueError("edges must not contain self-loops")
    if np.any((u < 0) | (u >= n) | (v < 0) | (v >= n)):
        raise ValueError("edge endpoints out of range [0, n)")
    key = np.minimum(u, v).astype(np.int64) * n + np.maximum(u, v).astype(np.int64)
    if np.unique(key).size != m:
        raise ValueError("duplicate undirected edges are not allowed (merge them first)")

    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    adj = sparse.csc_matrix((np.ones(2 * m), (rows, cols)), shape=(n, n))
    # +I guarantees every diagonal slot is stored (isolated nodes included).
    m_pattern = (lap(adj) + sparse.identity(n)).tocsc()
    m_pattern.sort_indices()
    indptr = m_pattern.indptr.astype(np.int64)
    indices = m_pattern.indices.astype(np.int64)
    nnz = m_pattern.nnz

    # (row, col) -> CSC position (one-time symbolic cost; not on the hot path).
    cols_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    pos = {(int(indices[p]), int(cols_of[p])): p for p in range(nnz)}
    diag_pos = np.array([pos[(i, i)] for i in range(n)], dtype=np.int64)
    off0 = np.empty(m, dtype=np.int64)
    off1 = np.empty(m, dtype=np.int64)
    ediag0 = np.empty(m, dtype=np.int64)
    ediag1 = np.empty(m, dtype=np.int64)
    for e in range(m):
        uu = int(u[e])
        vv = int(v[e])
        off0[e] = pos[(uu, vv)]
        off1[e] = pos[(vv, uu)]
        ediag0[e] = pos[(uu, uu)]
        ediag1[e] = pos[(vv, vv)]
    return m_pattern, diag_pos, off0, off1, ediag0, ediag1


class PURCLaplacianSolver:
    """Reusable PURC inner solver: fixed incidence, per-iteration weights/eps/RHS."""

    def __init__(self, graph, n_nodes=None, *, config: Optional[SolverConfig] = None) -> None:
        """
        Cache the incidence pattern and build the batched backend once.

        Args:
            graph: The network as either an ``(m, 2)`` **integer edge list**
                (undirected endpoints, 0-based), a **square adjacency / Laplacian /
                SDDM matrix** (edges = off-diagonal nonzeros), or a rectangular
                ``n x m`` **incidence ``C``** (edges = each column's two endpoints).
                Matrix forms may be scipy sparse / COO, numpy dense, or a CPU torch
                tensor. Must describe a simple graph (no self-loops, no duplicate
                edges).
            n_nodes: Number of network nodes ``n``; required for an edge list unless
                inferable from it, and validated against the matrix shape otherwise.
            config: Solver configuration (selects the route via ``config.method``).

        Raises:
            ValueError: If ``graph`` / ``n_nodes`` is malformed.

        """
        edges, self._n = _to_edges_and_n(graph, n_nodes)
        self._m = int(edges.shape[0])
        self.config = config or SolverConfig()
        (m_pattern, self._diag_pos, self._off0, self._off1, self._ediag0, self._ediag1) = (
            _build_pattern_and_maps(edges, self._n)
        )
        self._nnz = int(m_pattern.nnz)
        self._indptr = m_pattern.indptr.astype(np.int64)
        self._indices = m_pattern.indices.astype(np.int64)
        self._solver = BatchedSDDMSolver(m_pattern, config=self.config)
        self.method = self._solver.method

    @classmethod
    def from_incidence(cls, incidence, *, config: Optional[SolverConfig] = None):
        """
        Build from a node-link incidence matrix ``C`` (``n x m``).

        Unlike the constructor's shape heuristic, this always parses ``incidence``
        as an incidence (each column = one edge's two endpoints), so it is the
        unambiguous choice for a *square* incidence (e.g. a cycle, where ``m == n``).

        Args:
            incidence: The ``n x m`` incidence (scipy sparse / dense / CPU torch);
                each column has exactly two nonzeros.
            config: Solver configuration.

        Returns:
            A :class:`PURCLaplacianSolver`.

        """
        mat = adjacency_to_csc(incidence)
        mat.eliminate_zeros()
        coo = mat.tocoo()
        edges = _incidence_to_edges(coo)
        return cls(edges, int(coo.shape[0]), config=config)

    @classmethod
    def from_adjacency(cls, adjacency, *, config: Optional[SolverConfig] = None):
        """
        Build from a (symmetric) adjacency / Laplacian / SDDM matrix (``n x n``).

        Edges are the off-diagonal nonzero positions (deduplicated across the two
        triangles).

        Args:
            adjacency: The ``n x n`` matrix (scipy sparse / dense / CPU torch).
            config: Solver configuration.

        Returns:
            A :class:`PURCLaplacianSolver`.

        """
        mat = adjacency_to_csc(adjacency)
        mat.eliminate_zeros()
        coo = mat.tocoo()
        sel = coo.row != coo.col
        u = np.minimum(coo.row[sel], coo.col[sel])
        v = np.maximum(coo.row[sel], coo.col[sel])
        edges = np.unique(np.stack([u, v], axis=1).astype(np.int64), axis=0)
        return cls(edges, int(coo.shape[0]), config=config)

    @property
    def n(self) -> int:
        """
        Number of network nodes.

        Returns:
            The system dimension ``n``.

        """
        return self._n

    @property
    def num_edges(self) -> int:
        """
        Number of undirected edges.

        Returns:
            The edge count ``m``.

        """
        return self._m

    @property
    def phase(self) -> str:
        """
        The structural regime of the solve route.

        Returns:
            ``"forest"`` if the network is acyclic (exact zero-fill), else
            ``"cyclic"`` (direct CHOLMOD).

        """
        return "forest" if self.method == "forest" else "cyclic"

    def _assemble(self, weights_2d: np.ndarray, eps_1d: np.ndarray) -> np.ndarray:
        """
        Assemble ``M = C diag(w) C^T + eps*I`` values for each system natively.

        Args:
            weights_2d: ``[B, m]`` effective edge weights (mask already folded in).
            eps_1d: ``[B]`` per-system regularizer.

        Returns:
            A ``[B, nnz]`` float64 array of CSC values in the cached pattern order.

        """
        batch = weights_2d.shape[0]
        values = np.empty((batch, self._nnz), dtype=np.float64)
        native_core().assemble_sddm_values_f64(
            self._diag_pos,
            self._off0,
            self._off1,
            self._ediag0,
            self._ediag1,
            np.ascontiguousarray(weights_2d, dtype=np.float64),
            np.ascontiguousarray(eps_1d, dtype=np.float64),
            values,
        )
        return values

    def _effective_weights(self, weights, active_mask):
        """
        Fold an optional active mask into the edge weights.

        Args:
            weights: ``[m]`` or ``[B, m]`` edge weights (numpy / torch).
            active_mask: Optional ``[m]`` or ``[B, m]`` mask (1 = active).

        Returns:
            The effective weights as a contiguous float64 numpy array (same ndim
            as the broadcast of ``weights`` and ``active_mask``).

        Raises:
            ValueError: If the trailing dimension is not the edge count ``m``.

        """
        w = np.ascontiguousarray(VectorAdapter(weights).array, dtype=np.float64)
        if w.shape[-1] != self._m:
            raise ValueError(f"weights last dim {w.shape[-1]} != number of edges m={self._m}")
        if active_mask is not None:
            mask = np.ascontiguousarray(VectorAdapter(active_mask).array, dtype=np.float64)
            if mask.shape[-1] != self._m:
                raise ValueError(f"active_mask last dim {mask.shape[-1]} != m={self._m}")
            w = w * mask
        return w

    def _eps_vector(self, eps, batch: int) -> np.ndarray:
        """
        Coerce ``eps`` to a length-``batch`` float64 vector.

        Args:
            eps: A scalar or length-``batch`` array of regularizers.
            batch: The number of systems B.

        Returns:
            A contiguous ``[B]`` float64 array.

        Raises:
            ValueError: If ``eps`` is neither scalar nor length ``batch``.

        """
        arr = np.asarray(eps, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.full(batch, float(arr[0]), dtype=np.float64)
        if arr.size != batch:
            raise ValueError(f"eps must be a scalar or length B={batch}, got size {arr.size}")
        return np.ascontiguousarray(arr)

    def _result(self, solution) -> dict:
        """
        Wrap a solution in a PUM-style result dict.

        Args:
            solution: The solved Newton step(s).

        Returns:
            A dict with ``solution``, ``method``, ``phase``, ``converged``,
            ``iterations``.

        """
        return {
            "solution": solution,
            "method": self.method,
            "phase": self.phase,
            "converged": True,
            "iterations": 0,
        }

    def solve_batch(self, weights, *, active_mask=None, eps, rhs, rhs_offsets=None, warm_start=None):
        """
        Assemble and solve a batch of PURC Newton systems in one native call.

        Args:
            weights: Per-edge weights ``D`` - ``[m]`` (one matrix shared across the
                batch) or ``[B, m]`` (one matrix per system).
            active_mask: Optional active-edge mask, ``[m]`` or ``[B, m]`` (folded
                into the weights; inactive edges contribute zero).
            eps: The regularizer ``eps_k`` - a scalar or a ``[B]`` array.
            rhs: ``[B, n]`` (one RHS per system, or B RHS for a shared matrix) or,
                with ``rhs_offsets``, a flat ``[totalK, n]`` block.
            rhs_offsets: Optional int64 ``[B+1]`` offsets for ragged RHS per system.
            warm_start: Ignored (the direct routes need no warm start; accepted
                for interface symmetry with iterative solvers).

        Returns:
            A PUM-style dict ``{"solution", "method", "phase", "converged",
            "iterations"}``; the solution matches ``rhs``'s array kind / dtype /
            device.

        """
        del warm_start  # direct solve: no warm start needed
        w = self._effective_weights(weights, active_mask)
        shared = w.ndim == 1
        if not shared and w.ndim != 2:
            raise ValueError("weights must be 1-D [m] or 2-D [B, m]")
        if shared:
            eps_v = self._eps_vector(eps, 1)
            values = self._assemble(w[None, :], eps_v)[0]  # [nnz] shared matrix
        else:
            eps_v = self._eps_vector(eps, w.shape[0])
            values = self._assemble(w, eps_v)  # [B, nnz]
        solution = self._solver.solve_batch(values, rhs, rhs_offsets=rhs_offsets)
        return self._result(solution)

    def solve_step(self, weights, *, active_mask=None, eps, rhs, warm_start=None):
        """
        Assemble and solve a single PURC Newton system.

        Args:
            weights: ``[m]`` per-edge weights ``D``.
            active_mask: Optional ``[m]`` active-edge mask.
            eps: Scalar regularizer ``eps_k``.
            rhs: ``[n]`` right-hand side ``-r^k``.
            warm_start: Ignored (accepted for interface symmetry).

        Returns:
            A PUM-style dict whose ``solution`` is the length-``n`` Newton step in
            ``rhs``'s array kind.

        Raises:
            ValueError: If ``weights`` or ``rhs`` is not 1-dimensional.

        """
        w = np.asarray(VectorAdapter(weights).array)
        if w.ndim != 1:
            raise ValueError("solve_step weights must be 1-D [m]; use solve_batch for batches")
        adapter = VectorAdapter(rhs)
        if adapter.array.ndim != 1:
            raise ValueError("solve_step rhs must be 1-D [n]; use solve_batch for batches")
        rhs2d = adapter.restore(adapter.array[None, :])
        result = self.solve_batch(
            weights, active_mask=active_mask, eps=eps, rhs=rhs2d, warm_start=warm_start
        )
        result["solution"] = result["solution"][0]
        return result
