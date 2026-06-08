"""
One-shot dispatch entry points.

Convenience wrappers that build a :class:`~purc.laplaciansolve.solver.LaplacianSolver`
handle and solve in a single call, accepting numpy / scipy / torch (CPU) inputs
and returning the solution in the same array kind.  For repeated solves on the
same graph, construct a :class:`LaplacianSolver` directly to reuse the cached
preconditioner.
"""

from __future__ import annotations

from typing import Optional

from .solver import LaplacianSolver, SolverConfig


def solve(a, b, *, config: Optional[SolverConfig] = None, **config_kwargs):
    """
    Solve the Laplacian system ``lap(a) x = b - mean(b)`` in one call.

    Args:
        a: Adjacency matrix (numpy / scipy / torch; CPU).
        b: Right-hand side, ``(n,)`` or batched ``(B, n)`` (numpy / scipy / torch).
        config: Solver configuration; if ``None``, built from ``config_kwargs``.
        **config_kwargs: Forwarded to :class:`SolverConfig` when ``config`` is ``None``.

    Returns:
        The solution, in the same array kind / dtype / device as ``b``.  A 1-D
        ``b`` returns a 1-D solution; a 2-D ``b`` returns ``(B, n)``.

    """
    cfg = config or SolverConfig(**config_kwargs)
    handle = LaplacianSolver(a, config=cfg)
    import numpy as np

    from ._convert import is_torch

    ndim = b.ndim if (is_torch(b) or hasattr(b, "ndim")) else np.asarray(b).ndim
    return handle.solve_batch(b) if ndim == 2 else handle.solve(b)
