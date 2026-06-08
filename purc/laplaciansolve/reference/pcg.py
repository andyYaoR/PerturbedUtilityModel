"""
Preconditioned conjugate gradient (PCG) and plain CG.

Port of ``pcg`` / ``cg`` from ``Laplacians.jl`` (``src/pcg.jl``), including the
"best residual" iterate tracking and the ``stag_test`` stagnation guard.  The
preconditioner is supplied as a callable ``pre(r) -> z`` (typically the
:func:`~purc.laplaciansolve.reference.ldl_solve.ldl_solve` closure).
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional

import numpy as np

_EPS = float(np.finfo(np.float64).eps)


class PCGResult(NamedTuple):
    """
    Result of a (P)CG solve.

    Attributes:
        x: Best-residual solution iterate.
        iterations: Number of iterations performed.
        relres: Final relative residual ``‖b - mat x‖ / ‖b‖`` at the best iterate.
        converged: Whether ``relres`` reached ``tol``.

    """

    x: np.ndarray
    iterations: int
    relres: float
    converged: bool


def _matvec(mat, p: np.ndarray) -> np.ndarray:
    """
    Apply ``mat`` to vector ``p`` (sparse or callable).

    Args:
        mat: A SciPy sparse matrix or a callable ``p -> mat @ p``.
        p: The vector to multiply.

    Returns:
        The product ``mat @ p``.

    """
    if callable(mat):
        return mat(p)
    return mat @ p


def pcg(
    mat,
    b: np.ndarray,
    pre: Callable[[np.ndarray], np.ndarray],
    *,
    tol: float = 1e-6,
    maxits: int = 1000,
    stag_test: int = 5,
    x0: Optional[np.ndarray] = None,
) -> PCGResult:
    """
    Solve the symmetric system ``mat x = b`` with preconditioner ``pre``.

    Args:
        mat: Symmetric PSD/SPD operator (sparse matrix or callable matvec).
        b: Right-hand side vector.
        pre: Preconditioner applying ``M^{-1}`` to a residual: ``pre(r) -> z``.
        tol: Target relative residual ``‖b - mat x‖ / ‖b‖``.
        maxits: Maximum iterations.
        stag_test: Stagnation window ``k``; stop if ``rho`` fails to drop below
            ``(1 - 1/k)`` of its best for more than ``k`` consecutive iterations.
            ``0`` disables the test.
        x0: Optional warm-start iterate (defaults to zeros).  The residual is
            recomputed from ``x0``.

    Returns:
        A :class:`PCGResult` with the best-residual iterate and diagnostics.

    """
    b = np.asarray(b, dtype=np.float64)
    n = b.shape[0]
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return PCGResult(np.zeros(n), 0, 0.0, True)

    if x0 is None:
        x = np.zeros(n)
        r = b.copy()
    else:
        x = np.array(x0, dtype=np.float64, copy=True)
        r = b - _matvec(mat, x)

    bestx = x.copy()
    bestnr = float(np.linalg.norm(r)) / nb

    z = pre(r)
    p = z.copy()
    rho = float(np.dot(r, z))
    best_rho = rho
    stag_count = 0

    itcnt = 0
    converged = bestnr < tol
    while itcnt < maxits and not converged:
        itcnt += 1

        q = _matvec(mat, p)
        pq = float(np.dot(p, q))
        # On a PSD operator pq = p^T A p >= 0; pq <= 0 means the search direction
        # has reached the null space (converged/degenerate), so stop before
        # dividing.  Julia tolerates this via float Inf semantics; we guard it.
        if not math.isfinite(pq) or pq <= 0.0:
            break
        al = rho / pq

        # Periodic stagnation heuristics (cheap; only every 10th iteration).
        if itcnt % 10 == 0 and al * float(np.abs(p).sum()) < _EPS * float(np.abs(x).sum()):
            break

        x += al * p

        if itcnt % 10 == 0 and al * float(np.abs(q).sum()) < _EPS * float(np.abs(r).sum()):
            break

        r -= al * q

        nr = float(np.linalg.norm(r)) / nb
        if nr < bestnr:
            bestnr = nr
            bestx = x.copy()
        if nr < tol:
            converged = True
            break

        z = pre(r)
        oldrho = rho
        rho = float(np.dot(z, r))

        # stag_test: track the best rho and bail if it stalls.
        if stag_test > 0:
            threshold = best_rho * (1.0 - 1.0 / stag_test)
            if rho < threshold:
                best_rho = rho
                stag_count = 0
            elif best_rho > (1.0 - 1.0 / stag_test) * rho:
                stag_count += 1
                if stag_count > stag_test:
                    break

        if math.isinf(rho):
            break

        if oldrho == 0.0:  # previous rho collapsed to zero => converged
            break
        beta = rho / oldrho
        if math.isinf(beta):
            break

        if itcnt % 10 == 0 and beta * float(np.abs(p).sum()) < _EPS * float(np.abs(z).sum()):
            break

        p = z + beta * p

    return PCGResult(bestx, itcnt, bestnr, bestnr < tol)


def cg(
    mat,
    b: np.ndarray,
    *,
    tol: float = 1e-6,
    maxits: int = 1000,
    x0: Optional[np.ndarray] = None,
) -> PCGResult:
    """
    Solve the symmetric system ``mat x = b`` with unpreconditioned CG.

    Args:
        mat: Symmetric PSD/SPD operator (sparse matrix or callable matvec).
        b: Right-hand side vector.
        tol: Target relative residual.
        maxits: Maximum iterations.
        x0: Optional warm-start iterate (defaults to zeros).

    Returns:
        A :class:`PCGResult` with the best-residual iterate and diagnostics.

    """
    return pcg(mat, b, lambda r: r, tol=tol, maxits=maxits, stag_test=0, x0=x0)
