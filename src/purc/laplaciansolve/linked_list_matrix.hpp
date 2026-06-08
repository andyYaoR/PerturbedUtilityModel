// Adaptive linked-list matrix for the approxChol :deg elimination.
//
// Port of LLp / LLmatp + get_ll_col + compressCol! + flipIndex from
// Laplacians.jl (src/approxChol.jl, graphUtils.jl).  The Julia object-pointer
// linked list is represented as a struct-of-arrays node pool indexed by the
// original CSC nonzero position: next[p] == p terminates a column, reverse[p]
// is the twin entry of the same undirected edge.  Stable sorts are used so the
// build matches the Python/Julia reference bit-for-bit under a shared RNG.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "approx_chol_pq.hpp"

namespace lsolve {

// Reverse-edge map: flip[p] is the CSC position of the transpose of entry p.
// Requires row indices sorted within each column (scipy sort_indices), so the
// twin of (row r, col c) is found by binary search for c in column r - an
// O(nnz log deg) transpose, matching Julia's flipIndex without hashing.
template <typename Tind>
std::vector<Tind> compute_flip_index(const Tind* indptr, const Tind* indices,
                                     std::size_t n, std::size_t nnz) {
  std::vector<Tind> col_of(nnz);
  for (std::size_t j = 0; j < n; ++j) {
    for (Tind p = indptr[j]; p < indptr[j + 1]; ++p) {
      col_of[p] = static_cast<Tind>(j);
    }
  }
  std::vector<Tind> flip(nnz);
  for (std::size_t p = 0; p < nnz; ++p) {
    const Tind r = indices[p];   // row of entry p
    const Tind c = col_of[p];    // column of entry p
    const Tind lo = indptr[r];
    const Tind hi = indptr[r + 1];
    const Tind* found = std::lower_bound(indices + lo, indices + hi, c);
    flip[p] = static_cast<Tind>(found - indices);
  }
  return flip;
}

template <typename Tind, typename Tval>
struct LLmatp {
  Tind n;
  std::vector<Tind> degs;     // initial degree per vertex
  std::vector<Tind> cols;     // head node index per column (-1 if empty)
  std::vector<Tind> row;      // node pool: row index
  std::vector<Tval> val;      // node pool: edge weight (0 == deleted)
  std::vector<Tind> next;     // node pool: next-in-column (== self terminates)
  std::vector<Tind> reverse;  // node pool: twin entry
};

template <typename Tind, typename Tval>
LLmatp<Tind, Tval> build_llmatp(const Tind* indptr, const Tind* indices,
                                const Tval* data, std::size_t n) {
  const std::size_t nnz = static_cast<std::size_t>(indptr[n]);
  LLmatp<Tind, Tval> m;
  m.n = static_cast<Tind>(n);
  m.degs.resize(n);
  m.cols.assign(n, ApproxCholPQ<Tind>::kNone);
  m.row.resize(nnz);
  m.val.resize(nnz);
  m.next.resize(nnz);
  m.reverse.resize(nnz);

  for (std::size_t j = 0; j < n; ++j) {
    const Tind start = indptr[j];
    const Tind end = indptr[j + 1];
    m.degs[j] = end - start;
    if (end == start) {
      continue;
    }
    m.row[start] = indices[start];
    m.val[start] = data[start];
    m.next[start] = start;  // self-terminating tail
    Tind nxt = start;
    for (Tind p = start + 1; p < end; ++p) {
      m.row[p] = indices[p];
      m.val[p] = data[p];
      m.next[p] = nxt;
      nxt = p;
    }
    m.cols[j] = nxt;  // head is the last-created node
  }

  const std::vector<Tind> flip = compute_flip_index<Tind>(indptr, indices, n, nnz);
  for (std::size_t p = 0; p < nnz; ++p) {
    m.reverse[p] = flip[p];
  }
  return m;
}

// Gather the live (val > 0) node indices of column i, in list order.
template <typename Tind, typename Tval>
void get_ll_col(const LLmatp<Tind, Tval>& a, Tind i, std::vector<Tind>& out) {
  out.clear();
  Tind ll = a.cols[i];
  while (a.next[ll] != ll) {
    if (a.val[ll] > Tval(0)) {
      out.push_back(ll);
    }
    ll = a.next[ll];
  }
  if (a.val[ll] > Tval(0)) {
    out.push_back(ll);
  }
}

// Merge duplicate-row entries (summing weights, zeroing twins, decrementing the
// merged neighbour's degree), then stable-sort the result by weight ascending.
// `compressed` receives the compacted node indices.
template <typename Tind, typename Tval>
void compress_col(LLmatp<Tind, Tval>& a, std::vector<Tind>& colspace,
                  ApproxCholPQ<Tind>& pq, std::vector<Tind>& compressed) {
  std::stable_sort(colspace.begin(), colspace.end(),
                   [&](Tind x, Tind y) { return a.row[x] < a.row[y]; });
  compressed.clear();
  for (Tind idx : colspace) {
    if (!compressed.empty() && a.row[compressed.back()] == a.row[idx]) {
      a.val[compressed.back()] += a.val[idx];
      a.val[a.reverse[idx]] = Tval(0);
      pq.dec(a.row[compressed.back()]);
    } else {
      compressed.push_back(idx);
    }
  }
  std::stable_sort(compressed.begin(), compressed.end(),
                   [&](Tind x, Tind y) { return a.val[x] < a.val[y]; });
}

}  // namespace lsolve
