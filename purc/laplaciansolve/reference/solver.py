"""
High-level reference solvers: ``approxchol_lap`` and ``approxchol_sddm``.

These mirror the public entry points of ``Laplacians.jl``: each returns a closure
``f(b) -> x`` built once from the matrix.  ``approxchol_lap`` solves Laplacian
systems of an adjacency matrix (preconditioned CG against ``lap(a)`` with the
approxChol :class:`LDLinv` as preconditioner), splitting disconnected graphs by
component; ``approxchol_sddm`` solves SDDM systems via the grounding embedding.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import sparse

from .approx_chol import approx_chol
from .components import components, submatrix, vec_to_comps
from .graph import lap, to_csc, validate_adjacency
from .ldl_solve import ldl_solve
from .params import ApproxCholParams
from .pcg import PCGResult, pcg
from .rng import SampleStream, as_stream
from .sddm import sddm_wrap_lap


def _connected_solver(
    a: sparse.csc_matrix,
    *,
    tol: float,
    maxits: int,
    stag_test: int,
    order: str,
    rng: SampleStream,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a PCG solver closure for a single connected component.

    Args:
        a: Connected adjacency matrix (CSC).
        tol: PCG relative-residual tolerance.
        maxits: PCG iteration cap.
        stag_test: PCG stagnation window.
        order: Elimination ordering for the preconditioner build.
        rng: Sample stream driving the approxChol build.

    Returns:
        A closure ``f(b) -> x`` solving ``lap(a) x = b - mean(b)``.

    """
    la = lap(a)
    ldli = approx_chol(a, ApproxCholParams(order=order, stag_test=stag_test), rng=rng)

    def precond(r: np.ndarray) -> np.ndarray:
        return ldl_solve(ldli, r)

    def solve(b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64)
        result = pcg(la, b - b.mean(), precond, tol=tol, maxits=maxits, stag_test=stag_test)
        return result.x

    return solve


def approxchol_lap(
    a,
    *,
    tol: float = 1e-6,
    maxits: int = 1000,
    stag_test: int = 5,
    order: str = "deg",
    seed: Optional[int] = None,
    rng: SampleStream | int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a solver for Laplacian systems of the adjacency matrix *a*.

    Args:
        a: Symmetric, nonnegative, zero-diagonal adjacency matrix.
        tol: PCG relative-residual tolerance.
        maxits: PCG iteration cap.
        stag_test: PCG stagnation window (``0`` disables).
        order: Elimination ordering (currently ``"deg"``).
        seed: Convenience seed for the build RNG (used when ``rng`` is ``None``).
        rng: Explicit sample stream; overrides ``seed`` when given.

    Returns:
        A closure ``f(b) -> x`` solving ``lap(a) x = b - mean(b)`` per connected
        component (singleton components return zeros).

    """
    a = to_csc(a)
    validate_adjacency(a)
    stream = as_stream(rng if rng is not None else seed)

    n_comp, labels = components(a)
    if n_comp == 1:
        return _connected_solver(
            a, tol=tol, maxits=maxits, stag_test=stag_test, order=order, rng=stream
        )

    comps = vec_to_comps(labels)
    solvers: list[Optional[Callable[[np.ndarray], np.ndarray]]] = []
    for idx in comps:
        if idx.size == 1:
            solvers.append(None)  # singleton: nullSolver (returns zeros)
        else:
            asub = submatrix(a, idx)
            solvers.append(
                _connected_solver(
                    asub, tol=tol, maxits=maxits, stag_test=stag_test, order=order, rng=stream
                )
            )

    def solve(b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.float64)
        x = np.zeros_like(b)
        for idx, comp_solver in zip(comps, solvers):
            if comp_solver is not None:
                x[idx] = comp_solver(b[idx])
        return x

    return solve


def approxchol_sddm(
    m,
    *,
    tol: float = 1e-6,
    maxits: int = 1000,
    stag_test: int = 5,
    order: str = "deg",
    seed: Optional[int] = None,
    rng: SampleStream | int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a solver for SDDM systems ``M x = b``.

    Args:
        m: A symmetric SDDM matrix (e.g. the PURC ``H + eps*I``).
        tol: PCG relative-residual tolerance.
        maxits: PCG iteration cap.
        stag_test: PCG stagnation window.
        order: Elimination ordering for the preconditioner build.
        seed: Convenience seed for the build RNG (used when ``rng`` is ``None``).
        rng: Explicit sample stream; overrides ``seed`` when given.

    Returns:
        A closure ``f(b) -> x`` solving ``M x = b``.

    """
    stream = as_stream(rng if rng is not None else seed)

    def lap_factory(adjacency) -> Callable[[np.ndarray], np.ndarray]:
        return approxchol_lap(
            adjacency, tol=tol, maxits=maxits, stag_test=stag_test, order=order, rng=stream
        )

    return sddm_wrap_lap(lap_factory)(m)


def approxchol_lap_pcg(
    a,
    b: np.ndarray,
    *,
    tol: float = 1e-6,
    maxits: int = 1000,
    stag_test: int = 5,
    order: str = "deg",
    seed: Optional[int] = None,
    rng: SampleStream | int | None = None,
) -> PCGResult:
    """
    Solve a single connected Laplacian system and return full PCG diagnostics.

    Convenience wrapper used by tests/benchmarks that need the iteration count
    and residual (the closures from :func:`approxchol_lap` return only ``x``).

    Args:
        a: Connected, symmetric, nonnegative, zero-diagonal adjacency matrix.
        b: Right-hand side (mean-centered internally).
        tol: PCG relative-residual tolerance.
        maxits: PCG iteration cap.
        stag_test: PCG stagnation window.
        order: Elimination ordering.
        seed: Convenience seed for the build RNG (used when ``rng`` is ``None``).
        rng: Explicit sample stream; overrides ``seed`` when given.

    Returns:
        The :class:`PCGResult` for ``lap(a) x = b - mean(b)``.

    """
    a = to_csc(a)
    validate_adjacency(a)
    stream = as_stream(rng if rng is not None else seed)
    la = lap(a)
    ldli = approx_chol(a, ApproxCholParams(order=order, stag_test=stag_test), rng=stream)
    b = np.asarray(b, dtype=np.float64)
    return pcg(
        la, b - b.mean(), lambda r: ldl_solve(ldli, r), tol=tol, maxits=maxits, stag_test=stag_test
    )
