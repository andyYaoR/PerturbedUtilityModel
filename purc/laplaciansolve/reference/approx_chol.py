"""
Approximate-Cholesky factorization (the ``:deg`` adaptive build).

Faithful 0-based port of the data structures and the ``approxChol(a::LLmatp)``
routine from ``Laplacians.jl`` (``src/approxChol.jl``): the singly-linked column
storage :class:`LLp` / :class:`LLmatp`, the bucketed degree priority queue
:class:`ApproxCholPQ`, the column gather/compress helpers, and the edge-by-edge
random elimination that produces an :class:`LDLinv`.

The build eliminates one vertex at a time (lowest current degree first).  For
each non-final edge of the eliminated column it samples a single survivor edge
weighted by the column's cumulative weight and records the elimination
multiplier; this is what makes the factorization an *approximate* preconditioner.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy import sparse

from .graph import flip_index, to_csc
from .ldl_solve import LDLinv
from .params import ApproxCholParams
from .rng import SampleStream, as_stream

# Sentinel for "no node" in the priority queue (Julia uses 0 with 1-based nodes).
_NONE = -1


class LLp:
    """
    A single linked-list entry of one column during elimination.

    Mirrors Julia's ``LLp``: ``next`` points to the next entry in the same column
    and points to *itself* to terminate the list; ``reverse`` points to the twin
    entry storing the same edge from the other endpoint's column.

    Attributes:
        row: Row index of this entry.
        val: Edge weight (set to ``0.0`` to logically delete the entry).
        next: Next entry in the column (``self`` marks the tail).
        reverse: The twin entry for this edge in the other column.

    """

    __slots__ = ("row", "val", "next", "reverse")

    def __init__(
        self,
        row: int = 0,
        val: float = 0.0,
        nxt: Optional["LLp"] = None,
        reverse: Optional["LLp"] = None,
    ) -> None:
        """
        Create an entry; ``next``/``reverse`` default to ``self``.

        Args:
            row: Row index.
            val: Edge weight.
            nxt: Next entry in the column, or ``None`` to self-terminate.
            reverse: Twin entry, or ``None`` to point at ``self`` for now.

        """
        self.row = row
        self.val = val
        self.next = nxt if nxt is not None else self
        self.reverse = reverse if reverse is not None else self


class LLmatp:
    """
    Adaptive linked-list matrix used during ``:deg`` elimination.

    Attributes:
        n: Number of vertices.
        degs: Initial degree of each vertex.
        cols: Per-column head entry (``cols[i]`` is the list head for column i).
        llelems: All entries, indexed by original CSC nonzero position.

    """

    __slots__ = ("n", "degs", "cols", "llelems")

    def __init__(
        self,
        n: int,
        degs: np.ndarray,
        cols: List[Optional[LLp]],
        llelems: List[LLp],
    ) -> None:
        """
        Store the prebuilt linked-list matrix fields.

        Args:
            n: Number of vertices.
            degs: Initial per-vertex degrees.
            cols: Per-column list heads.
            llelems: All list entries in CSC order.

        """
        self.n = n
        self.degs = degs
        self.cols = cols
        self.llelems = llelems


def build_llmatp(a: sparse.csc_matrix) -> LLmatp:
    """
    Build an :class:`LLmatp` from a symmetric adjacency matrix.

    Args:
        a: Symmetric, zero-diagonal adjacency matrix (CSC).

    Returns:
        The linked-list matrix with reverse-edge pointers wired up.

    """
    a = to_csc(a)
    n = a.shape[0]
    indptr = a.indptr
    indices = a.indices
    data = a.data
    m = a.nnz

    degs = np.diff(indptr).astype(np.int64)
    flips = flip_index(a)

    llelems: List[LLp] = [None] * m  # type: ignore[list-item]
    cols: List[Optional[LLp]] = [None] * n

    for i in range(n):
        start = int(indptr[i])
        end = int(indptr[i + 1])
        if end == start:
            continue  # isolated vertex; only reachable for non-connected input
        first = LLp(int(indices[start]), float(data[start]))  # self-terminating tail
        llelems[start] = first
        nxt = first
        for p in range(start + 1, end):
            entry = LLp(int(indices[p]), float(data[p]), nxt)
            llelems[p] = entry
            nxt = entry
        cols[i] = nxt  # head is the last-created entry

    for p in range(m):
        llelems[p].reverse = llelems[int(flips[p])]

    return LLmatp(n, degs, cols, llelems)


def get_ll_col(a: LLmatp, i: int) -> List[LLp]:
    """
    Gather the live (positive-weight) entries of column ``i``.

    Args:
        a: The linked-list matrix.
        i: Column (vertex) index.

    Returns:
        The column's entries with ``val > 0``, in list order.

    """
    colspace: List[LLp] = []
    ll = a.cols[i]
    while ll.next is not ll:
        if ll.val > 0.0:
            colspace.append(ll)
        ll = ll.next
    if ll.val > 0.0:  # the self-terminating tail entry
        colspace.append(ll)
    return colspace


def compress_col(colspace: List[LLp], pq: "ApproxCholPQ") -> List[LLp]:
    """
    Merge duplicate-row entries of a column, then sort by weight.

    Sorts by row, sums entries that share a row (zeroing the merged twin and
    decrementing the merged neighbor's degree), and finally sorts the compressed
    column by weight ascending (port of ``compressCol!``).

    Args:
        colspace: The gathered column entries.
        pq: Degree priority queue, decremented for each merged multi-edge.

    Returns:
        The compressed, weight-sorted column entries.

    """
    colspace.sort(key=lambda e: e.row)
    compressed: List[LLp] = []
    for entry in colspace:
        if compressed and compressed[-1].row == entry.row:
            compressed[-1].val += entry.val
            entry.reverse.val = 0.0
            pq.dec(compressed[-1].row)
        else:
            compressed.append(entry)
    compressed.sort(key=lambda e: e.val)
    return compressed


def _key_map(x: int, k: int, upper: int) -> int:
    """
    Bucket index for degree ``x`` (linear below ``k``, logarithmic above).

    Args:
        x: The degree (key) to bucket.
        k: Linear/logarithmic crossover (``split * n``).
        upper: Maximum bucket index.

    Returns:
        The bucket index for ``x``.

    """
    return x if x <= k else min(upper, k + x // k)


class ApproxCholPQ:
    """
    Approximate priority queue over vertex degrees (bucketed doubly-linked lists).

    Ports Julia's ``ApproxCholPQ``: vertices with (approximately) equal degree
    share a bucket, so ``pop`` returns a minimum-degree vertex and ``inc``/``dec``
    move a vertex between buckets only when its bucket index changes.
    """

    __slots__ = ("prev", "next", "key", "lists", "minlist", "nitems", "n", "split")

    def __init__(self, degs: np.ndarray, split: int = 1) -> None:
        """
        Initialize the queue from initial degrees.

        Args:
            degs: Initial per-vertex degree array.
            split: Multi-edge split factor (sizes the bucket array).

        """
        n = int(degs.shape[0])
        self.n = n
        self.split = int(split)
        self.prev = np.full(n, _NONE, dtype=np.int64)
        self.next = np.full(n, _NONE, dtype=np.int64)
        self.key = np.asarray(degs, dtype=np.int64).copy()
        # Bucket heads; index by key directly (== _key_map for the initial keys).
        self.lists = np.full(2 * self.split * n + 2, _NONE, dtype=np.int64)
        self.minlist = 1
        self.nitems = n

        for i in range(n):
            key = int(self.key[i])
            head = int(self.lists[key])
            if head >= 0:
                self.prev[i] = _NONE
                self.next[i] = head
                self.prev[head] = i
            else:
                self.prev[i] = _NONE
                self.next[i] = _NONE
            self.lists[key] = i

    def _upper(self) -> int:
        """
        Maximum bucket index.

        Returns:
            ``2 * split * n + 1``.

        """
        return 2 * self.split * self.n + 1

    def pop(self) -> int:
        """
        Remove and return a vertex of minimum (approximate) degree.

        Returns:
            The popped vertex index.

        Raises:
            RuntimeError: If the queue is empty.

        """
        if self.nitems == 0:
            raise RuntimeError("ApproxCholPQ is empty")
        while self.lists[self.minlist] == _NONE:
            self.minlist += 1
        i = int(self.lists[self.minlist])
        nxt = int(self.next[i])
        self.lists[self.minlist] = nxt
        if nxt >= 0:
            self.prev[nxt] = _NONE
        self.nitems -= 1
        return i

    def _move(self, i: int, newkey: int, oldlist: int, newlist: int) -> None:
        """
        Relocate vertex ``i`` from ``oldlist`` to the head of ``newlist``.

        Args:
            i: Vertex index.
            newkey: New degree key for ``i``.
            oldlist: Current bucket index.
            newlist: Target bucket index.

        """
        prev = int(self.prev[i])
        nxt = int(self.next[i])
        # Unlink from the old bucket.
        if nxt >= 0:
            self.prev[nxt] = prev
        if prev >= 0:
            self.next[prev] = nxt
        else:
            self.lists[oldlist] = nxt
        # Insert at the head of the new bucket.
        head = int(self.lists[newlist])
        if head >= 0:
            self.prev[head] = i
        self.lists[newlist] = i
        self.prev[i] = _NONE
        self.next[i] = head
        self.key[i] = newkey

    def dec(self, i: int) -> None:
        """
        Decrement the degree key of vertex ``i``.

        Args:
            i: Vertex index.

        """
        k = self.split * self.n
        upper = self._upper()
        old_key = int(self.key[i])
        oldlist = _key_map(old_key, k, upper)
        newlist = _key_map(old_key - 1, k, upper)
        if newlist != oldlist:
            self._move(i, old_key - 1, oldlist, newlist)
            if newlist < self.minlist:
                self.minlist = newlist
        else:
            self.key[i] = old_key - 1

    def inc(self, i: int) -> None:
        """
        Increment the degree key of vertex ``i``.

        Args:
            i: Vertex index.

        """
        k = self.split * self.n
        upper = self._upper()
        old_key = int(self.key[i])
        oldlist = _key_map(old_key, k, upper)
        newlist = _key_map(old_key + 1, k, upper)
        if newlist != oldlist:
            self._move(i, old_key + 1, oldlist, newlist)
        else:
            self.key[i] = old_key + 1


def _approx_chol_deg(a: LLmatp, rng: SampleStream) -> LDLinv:
    """
    Run the adaptive ``:deg`` edge-by-edge elimination.

    Args:
        a: The linked-list matrix to eliminate (mutated in place).
        rng: Uniform sample source driving the survivor-edge sampling.

    Returns:
        The resulting approximate factorization.

    """
    n = a.n
    col = np.zeros(max(n - 1, 0), dtype=np.int64)
    colptr = np.zeros(n, dtype=np.int64)
    rowval: List[int] = []
    fval: List[float] = []
    d = np.zeros(n, dtype=np.float64)

    pq = ApproxCholPQ(a.degs)

    it = 0
    while it < n - 1:
        i = pq.pop()

        col[it] = i
        colptr[it] = len(rowval)
        it += 1

        colspace = get_ll_col(a, i)
        colspace = compress_col(colspace, pq)
        length = len(colspace)

        vals = np.fromiter((e.val for e in colspace), dtype=np.float64, count=length)
        cumspace = np.cumsum(vals)
        csum = float(cumspace[-1])
        wdeg = csum
        col_scale = 1.0

        for joffset in range(length - 1):
            ll = colspace[joffset]
            w = ll.val * col_scale
            j = ll.row
            revj = ll.reverse

            f = w / wdeg

            # Sample one survivor edge weighted by the remaining cumulative mass.
            r = rng.rand() * (csum - float(cumspace[joffset])) + float(cumspace[joffset])
            koff = int(np.searchsorted(cumspace, r, side="left"))
            if koff >= length:
                koff = length - 1
            k = colspace[koff].row

            pq.inc(k)

            new_edge_val = f * (1.0 - f) * wdeg

            # Reuse revj as edge (j -> k) in column j; move ll into column k.
            revj.row = k
            revj.val = new_edge_val
            revj.reverse = ll

            khead = a.cols[k]
            a.cols[k] = ll
            ll.next = khead
            ll.reverse = revj
            ll.val = new_edge_val
            ll.row = j

            # Use plain multiplication (not ``** 2``) so the float arithmetic is
            # bit-identical to the C++ port (libm pow can differ by 1 ULP).
            one_minus_f = 1.0 - f
            col_scale = col_scale * one_minus_f
            wdeg = wdeg * (one_minus_f * one_minus_f)

            rowval.append(j)
            fval.append(f)

        # Final edge of the column carries the pivot weight into d[i].
        ll = colspace[length - 1]
        w = ll.val * col_scale
        j = ll.row
        revj = ll.reverse

        if it < n - 1:
            pq.dec(j)

        revj.val = 0.0

        rowval.append(j)
        fval.append(1.0)

        d[i] = w

    colptr[it] = len(rowval)

    return LDLinv(
        col=col,
        colptr=colptr,
        rowval=np.asarray(rowval, dtype=np.int64),
        fval=np.asarray(fval, dtype=np.float64),
        d=d,
    )


def approx_chol(
    a: sparse.csc_matrix,
    params: ApproxCholParams | None = None,
    rng: SampleStream | int | None = None,
) -> LDLinv:
    """
    Build an approximate-Cholesky factorization of ``lap(a)``.

    Args:
        a: Symmetric, nonnegative, zero-diagonal adjacency matrix (CSC).
        params: Elimination parameters; defaults to ``ApproxCholParams()``
            (``order="deg"``, no splitting).
        rng: A :class:`~purc.laplaciansolve.reference.rng.SampleStream`, an integer
            seed, or ``None`` for a fresh nondeterministic stream.

    Returns:
        The approximate factorization as an :class:`LDLinv`.

    Raises:
        NotImplementedError: If a not-yet-ported ordering or splitting is
            requested (the reference currently implements ``order="deg"`` with
            ``split == 0``).

    """
    params = params or ApproxCholParams()
    if params.order != "deg":
        raise NotImplementedError(
            f"reference approx_chol currently implements order='deg' only, "
            f"got order={params.order!r}"
        )
    if params.split != 0 or params.merge != 0:
        raise NotImplementedError("reference approx_chol currently implements split=merge=0 only")

    stream = as_stream(rng)
    llmat = build_llmatp(to_csc(a))
    return _approx_chol_deg(llmat, stream)
