"""
Native CPU solver backend.

Provides the same ``approxchol_lap`` / ``approxchol_sddm`` closures as the
reference, but with the approxChol build and the PCG solve executed in C++.  The
*graph preprocessing* (Laplacian assembly, connected-component splitting, the
SDDM grounding embedding, forest detection) reuses the shared reference
utilities; only the numerical linear solve runs natively.

For a forest the approxChol factorization is exact, so the closure applies it
once and skips PCG entirely (the acyclic fast path).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from scipy import sparse

from .._loader import native_core
from ..factorization import build_approxchol
from ..reference.components import components, submatrix, vec_to_comps
from ..reference.forest_ldl import is_forest
from ..reference.graph import lap, to_csc, validate_adjacency
from ..reference.sddm import sddm_wrap_lap


def _lap_csc_int64(a: sparse.csc_matrix):
    """
    Return ``lap(a)`` as a sorted CSC with int64 index arrays.

    Args:
        a: Adjacency matrix.

    Returns:
        ``(indptr, indices, data)`` for the native PCG matvec.

    """
    la = lap(a).tocsc()
    la.sort_indices()
    return la.indptr.astype(np.int64), la.indices.astype(np.int64), la.data.astype(np.float64)


def _connected_solver(
    a: sparse.csc_matrix,
    *,
    tol: float,
    maxits: int,
    stag_test: int,
    seed: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a native PCG solver closure for one connected component.

    Args:
        a: Connected adjacency matrix (CSC).
        tol: PCG relative-residual tolerance.
        maxits: PCG iteration cap.
        stag_test: PCG stagnation window.
        seed: Seed for the approxChol build RNG.

    Returns:
        A closure ``f(b) -> x`` solving ``lap(a) x = b - mean(b)``.

    """
    core = native_core()
    ldli = build_approxchol(a, seed=seed)
    n = a.shape[0]
    forest = is_forest(a)
    indptr, indices, data = _lap_csc_int64(a)

    def solve(b: np.ndarray) -> np.ndarray:
        b = np.ascontiguousarray(b, dtype=np.float64)
        bz = b - b.mean()
        if forest:
            # Exact factorization: one apply solves it, no PCG needed.
            return ldli.solve(bz, subtract_mean=True)
        x = np.zeros(n, dtype=np.float64)
        core.pcg_lap_f64(
            indptr,
            indices,
            data,
            bz,
            ldli.col,
            ldli.colptr,
            ldli.rowval,
            ldli.fval,
            ldli.d,
            x,
            tol,
            maxits,
            stag_test,
        )
        return x

    return solve


def approxchol_lap(
    a,
    *,
    tol: float = 1e-6,
    maxits: int = 1000,
    stag_test: int = 5,
    seed: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Native solver for Laplacian systems of the adjacency matrix *a*.

    Args:
        a: Symmetric, nonnegative, zero-diagonal adjacency matrix.
        tol: PCG relative-residual tolerance.
        maxits: PCG iteration cap.
        stag_test: PCG stagnation window.
        seed: Seed for the approxChol build RNG.

    Returns:
        A closure ``f(b) -> x`` solving ``lap(a) x = b - mean(b)`` per connected
        component (singletons return zeros).

    """
    a = to_csc(a)
    validate_adjacency(a)
    n_comp, labels = components(a)
    if n_comp == 1:
        return _connected_solver(a, tol=tol, maxits=maxits, stag_test=stag_test, seed=seed)

    comps = vec_to_comps(labels)
    solvers: List[Optional[Callable[[np.ndarray], np.ndarray]]] = []
    for idx in comps:
        if idx.size == 1:
            solvers.append(None)
        else:
            solvers.append(
                _connected_solver(
                    submatrix(a, idx), tol=tol, maxits=maxits, stag_test=stag_test, seed=seed
                )
            )

    def solve(b: np.ndarray) -> np.ndarray:
        b = np.ascontiguousarray(b, dtype=np.float64)
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
    seed: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Native solver for SDDM systems ``M x = b`` via the grounding embedding.

    Args:
        m: A symmetric SDDM matrix (e.g. the PURC ``H + eps*I``).
        tol: PCG relative-residual tolerance.
        maxits: PCG iteration cap.
        stag_test: PCG stagnation window.
        seed: Seed for the approxChol build RNG.

    Returns:
        A closure ``f(b) -> x`` solving ``M x = b``.

    """

    def lap_factory(adjacency) -> Callable[[np.ndarray], np.ndarray]:
        return approxchol_lap(adjacency, tol=tol, maxits=maxits, stag_test=stag_test, seed=seed)

    return sddm_wrap_lap(lap_factory)(m)
