"""
Loader for TNTP (Transportation Networks for Test Problems) ``_net`` files.

This is a helper for the example scripts. It parses a ``*_net.tntp`` link file 
into the node-arc incidence matrixand the per-link attributes (capacity, length, 
and free-flow travel time) that define a perturbed-utility forward problem.

Each link is a directed arc from ``init_node`` to ``term_node``; a two-way road is
stored as two opposite arcs. The incidence matrix carries ``+1`` at the tail of each
arc and ``-1`` at its head, following the demand convention used throughout the
package (``b[origin] = +1`` and ``b[destination] = -1``), so the loaded matrix can
be passed directly to :class:`purc.static_purc.constraints.GeneralPolytope`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class TNTPNetwork:
    """
    A parsed TNTP network.

    Attributes:
        A: ``(n_nodes, n_links)`` node-arc incidence (``+1`` tail, ``-1`` head).
        capacity: ``(n_links,)`` link capacity.
        length: ``(n_links,)`` link length.
        fftt: ``(n_links,)`` free-flow travel time.
        n_nodes: Number of nodes.
        n_links: Number of directed links.
        node_index: Original TNTP node id -> contiguous row index.

    """

    A: sp.csr_matrix
    capacity: np.ndarray
    length: np.ndarray
    fftt: np.ndarray
    n_nodes: int
    n_links: int
    node_index: dict


def load_net(path: str) -> TNTPNetwork:
    """
    Load a TNTP ``_net`` file into a :class:`TNTPNetwork`.

    Args:
        path: Path to the ``*_net.tntp`` file.

    Returns:
        The parsed network (incidence + per-link length and free-flow time).

    """
    init, term, capacity, length, fftt = [], [], [], [], []
    started = False
    for line in open(path):
        s = line.strip()
        if s.startswith("<END OF METADATA>"):
            started = True
            continue
        if not started or not s or s.startswith("~"):
            continue
        p = s.replace(";", "").split()
        if len(p) < 6:
            continue
        init.append(int(p[0]))  # TNTP column 1 = init_node
        term.append(int(p[1]))  # column 2 = term_node
        capacity.append(float(p[2]))  # column 3 = capacity
        length.append(float(p[3]))  # column 4 = length
        fftt.append(float(p[4]))  # column 5 = free-flow time

    init = np.asarray(init)
    term = np.asarray(term)
    nodes = np.unique(np.concatenate([init, term]))
    index = {int(v): i for i, v in enumerate(nodes)}
    n, m = len(nodes), len(init)
    ii = np.fromiter((index[int(x)] for x in init), dtype=np.int64, count=m)
    tt = np.fromiter((index[int(x)] for x in term), dtype=np.int64, count=m)
    data = np.concatenate([np.ones(m), -np.ones(m)])  # +1 at tail, -1 at head
    rows = np.concatenate([ii, tt])
    cols = np.tile(np.arange(m), 2)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, m))
    return TNTPNetwork(
        A, np.asarray(capacity), np.asarray(length), np.asarray(fftt), n, m, index
    )
