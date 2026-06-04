"""
Minimal TNTP (Transportation Networks for Test Problems) ``_net`` loader.

Parses a ``*_net.tntp`` link file into the node-arc incidence matrix and per-link
attributes (length, free-flow time) used to build a PURC forward problem.  Each
link is a directed arc ``init_node -> term_node``; two-way roads appear as two
arcs (parallel undirected edges).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class TNTPNetwork:
    """A parsed TNTP network."""

    A: sp.csr_matrix  # (n_nodes, n_links) node-arc incidence
    length: np.ndarray  # (n_links,) link length
    fftt: np.ndarray  # (n_links,) free-flow travel time
    n_nodes: int
    n_links: int
    node_index: dict  # original node id -> row index


def load_net(path: str) -> TNTPNetwork:
    """
    Load a TNTP ``_net`` file into a :class:`TNTPNetwork`.

    Args:
        path: Path to the ``*_net.tntp`` file.

    Returns:
        The parsed network.

    """
    init, term, length, fftt = [], [], [], []
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
        init.append(int(p[0]))
        term.append(int(p[1]))
        length.append(float(p[3]))
        fftt.append(float(p[4]))

    init = np.asarray(init)
    term = np.asarray(term)
    nodes = np.unique(np.concatenate([init, term]))
    index = {int(v): i for i, v in enumerate(nodes)}
    n, m = len(nodes), len(init)
    ii = np.fromiter((index[int(x)] for x in init), dtype=np.int64, count=m)
    tt = np.fromiter((index[int(x)] for x in term), dtype=np.int64, count=m)
    data = np.concatenate([np.ones(m), -np.ones(m)])
    rows = np.concatenate([ii, tt])
    cols = np.tile(np.arange(m), 2)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, m))
    return TNTPNetwork(A, np.asarray(length), np.asarray(fftt), n, m, index)
