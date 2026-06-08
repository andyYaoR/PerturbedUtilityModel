"""
Connected-component helpers.

Ports the role of ``components`` / ``vecToComps`` from ``Laplacians.jl``: a
Laplacian is singular once per connected component, so disconnected graphs are
solved component-by-component.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .graph import to_csc


def components(a) -> Tuple[int, np.ndarray]:
    """
    Return the connected components of the graph with adjacency *a*.

    Args:
        a: Symmetric adjacency matrix.

    Returns:
        A pair ``(n_components, labels)`` where ``labels[i]`` is the component
        index of vertex ``i``.

    """
    csc = to_csc(a)
    n_components, labels = csgraph.connected_components(csc, directed=False, connection="weak")
    return int(n_components), labels.astype(np.int64)


def vec_to_comps(labels: np.ndarray) -> List[np.ndarray]:
    """
    Group vertex indices by component label.

    Args:
        labels: Per-vertex component labels (as returned by :func:`components`).

    Returns:
        A list whose ``c``-th entry is the sorted array of vertex indices in
        component ``c``.

    """
    n_components = int(labels.max()) + 1 if labels.size else 0
    comps: List[np.ndarray] = [np.flatnonzero(labels == c) for c in range(n_components)]
    return comps


def submatrix(a: sparse.csc_matrix, idx: np.ndarray) -> sparse.csc_matrix:
    """
    Extract the principal submatrix ``a[idx, idx]``.

    Args:
        a: A sparse matrix.
        idx: Vertex indices to keep.

    Returns:
        The induced submatrix as CSC.

    """
    return to_csc(to_csc(a)[idx][:, idx])
