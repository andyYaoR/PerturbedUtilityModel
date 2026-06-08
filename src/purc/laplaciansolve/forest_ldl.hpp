/**
 * Exact zero-fill LDL^T for SDDM systems whose off-diagonal graph is a forest.
 *
 * At the optimum of the PURC Newton iteration the active subgraph is acyclic.  A
 * forest-structured SPD matrix M = (forest Laplacian) + diag factorizes with
 * *zero fill*: eliminate a degree-1 vertex v with surviving neighbor u, and the
 * only Schur update is to u's diagonal,
 *
 *     d_v       = M[v,v]   (already updated by v's eliminated children)
 *     L[u,v]    = M[u,v] / d_v
 *     M[u,u]   -= M[u,v]^2 / d_v
 *
 * so no new edges are ever created.  The whole factorization and each solve are
 * O(n); there is no ordering search and no library overhead, which beats a
 * general sparse Cholesky (CHOLMOD) in this regime.
 *
 * The pieces are split so they can be reused for a *batch* of matrices that share
 * one sparsity pattern (the PURC per-destination case):
 *   - ForestStructure  - the pattern-only elimination plan (order, parents, CSC
 *                         position maps, roots), built once by leaf-pruning;
 *                         detection and ordering are the same pass (if every
 *                         vertex eliminates, it is a forest; a surviving core of
 *                         degree>=2 means a cycle -> route to CHOLMOD).
 *   - forest_factor_into - numeric leaf elimination of one matrix's values into
 *                         caller-provided (mult, dinv) spans (dinv doubles as the
 *                         running-diagonal accumulator, so no extra scratch).
 *   - forest_solve_with  - forward / diagonal / backward sweep for k RHS.
 *   - ForestLDL          - single fixed-pattern matrix (build + update + solve).
 *   - ForestBatchSolver  - B matrices sharing the pattern, factored+solved in one
 *                         GIL-released call, parallel over systems, arenas reused.
 *
 * Edges are taken structurally (a stored off-diagonal is an edge); a numerically
 * zero edge factorizes harmlessly (it simply decouples its endpoints).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"

namespace lsolve {

/// Pattern-only elimination plan shared by every matrix with this sparsity.
template <typename Tind>
struct ForestStructure {
  Tind n = 0;
  std::size_t nnz = 0;
  bool is_forest = false;
  std::vector<Tind> order;            // eliminated vertices, leaves first
  std::vector<Tind> parent;           // surviving neighbor at elimination time
  std::vector<std::size_t> edge_pos;  // CSC data index of M[parent, v]
  std::vector<std::size_t> diag_pos;  // CSC data index of M[i, i] (npos if absent)
  std::vector<Tind> roots;            // vertices surviving all eliminations

  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  /// Leaf-pruning: builds the elimination order/parents/CSC maps and sets
  /// is_forest (true iff every vertex is eliminated).
  void build(const Tind* indptr, const Tind* indices, Tind n_in) {
    n = n_in;
    nnz = static_cast<std::size_t>(indptr[n]);
    diag_pos.assign(static_cast<std::size_t>(n), npos);
    std::vector<Tind> deg(static_cast<std::size_t>(n), 0);
    for (Tind j = 0; j < n; ++j) {
      for (Tind p = indptr[j]; p < indptr[j + 1]; ++p) {
        const Tind i = indices[p];
        if (i == j) {
          diag_pos[j] = static_cast<std::size_t>(p);
        } else {
          ++deg[j];
        }
      }
    }
    // Off-diagonal adjacency in CSR layout, each entry carrying its CSC position.
    std::vector<Tind> nbr_ptr(static_cast<std::size_t>(n) + 1, 0);
    for (Tind j = 0; j < n; ++j) nbr_ptr[j + 1] = nbr_ptr[j] + deg[j];
    std::vector<Tind> nbr(static_cast<std::size_t>(nbr_ptr[n]));
    std::vector<std::size_t> nbr_pos(static_cast<std::size_t>(nbr_ptr[n]));
    std::vector<Tind> cursor(nbr_ptr.begin(), nbr_ptr.end() - 1);
    for (Tind j = 0; j < n; ++j) {
      for (Tind p = indptr[j]; p < indptr[j + 1]; ++p) {
        const Tind i = indices[p];
        if (i == j) continue;
        const Tind c = cursor[j]++;
        nbr[c] = i;
        nbr_pos[c] = static_cast<std::size_t>(p);  // M[i, j] in column j (== M[j, i])
      }
    }

    std::vector<Tind> wdeg(deg);
    std::vector<char> alive(static_cast<std::size_t>(n), 1);
    std::vector<Tind> queue;
    queue.reserve(static_cast<std::size_t>(n));
    for (Tind j = 0; j < n; ++j) {
      if (wdeg[j] == 1) queue.push_back(j);
    }
    order.clear();
    parent.clear();
    edge_pos.clear();
    std::size_t head = 0;
    while (head < queue.size()) {
      const Tind v = queue[head++];
      if (!alive[v] || wdeg[v] != 1) continue;
      Tind u = -1;
      std::size_t epos = npos;
      for (Tind c = nbr_ptr[v]; c < nbr_ptr[v + 1]; ++c) {
        if (alive[nbr[c]]) {
          u = nbr[c];
          epos = nbr_pos[c];
          break;
        }
      }
      order.push_back(v);
      parent.push_back(u);
      edge_pos.push_back(epos);
      alive[v] = 0;
      wdeg[v] = 0;
      --wdeg[u];
      if (wdeg[u] == 1 && alive[u]) queue.push_back(u);
    }

    is_forest = true;
    for (Tind j = 0; j < n; ++j) {
      if (alive[j] && wdeg[j] >= 1) {
        is_forest = false;
        break;
      }
    }
    roots.clear();
    if (is_forest) {
      for (Tind j = 0; j < n; ++j) {
        if (alive[j]) roots.push_back(j);
      }
    }
  }
};

/**
 * Numeric leaf-elimination factorization of one matrix's `data` into the
 * caller-provided `mult` (length order.size()) and `dinv` (length n) spans.
 *
 * `dinv` doubles as the running-diagonal accumulator: each entry holds vertex
 * j's Schur-complemented diagonal until j is eliminated, at which point it is
 * overwritten in place with 1/pivot (j is never read as a diagonal again).  This
 * removes the need for any per-system scratch, so a batch can factor B systems
 * with only the B output arenas.
 */
template <typename Tind, typename Tval>
void forest_factor_into(const ForestStructure<Tind>& s, const Tval* data, Tval* mult,
                        Tval* dinv) {
  const Tind n = s.n;
  for (Tind j = 0; j < n; ++j) {
    dinv[j] = (s.diag_pos[j] != ForestStructure<Tind>::npos) ? data[s.diag_pos[j]] : Tval(0);
  }
  const std::size_t ne = s.order.size();
  for (std::size_t e = 0; e < ne; ++e) {
    const Tind v = s.order[e];
    const Tind u = s.parent[e];
    const Tval w = data[s.edge_pos[e]];  // M[u, v]
    const Tval piv = dinv[v];
    const Tval m = w / piv;
    mult[e] = m;
    dinv[u] -= w * m;        // Schur complement onto u's diagonal only
    dinv[v] = Tval(1) / piv;  // v eliminated: convert its accumulator to 1/pivot
  }
  for (Tind root : s.roots) dinv[root] = Tval(1) / dinv[root];
}

/**
 * Solve M X = B for k right-hand sides from a factorization (mult, dinv).
 *
 * b and x are flat length n*k buffers laid out [k, n] row-major (RHS r is the
 * slice [r*n, (r+1)*n)).  Serial over the k RHS - the batch parallelizes one
 * level up, over independent systems.
 */
template <typename Tind, typename Tval>
void forest_solve_with(const ForestStructure<Tind>& s, const Tval* mult, const Tval* dinv,
                       const Tval* b, Tval* x, Tind k) {
  const std::size_t ne = s.order.size();
  const Tind* ord = s.order.data();
  const Tind* par = s.parent.data();
  const Tind n = s.n;
  for (Tind r = 0; r < k; ++r) {
    const Tval* br = b + static_cast<std::size_t>(r) * static_cast<std::size_t>(n);
    Tval* xr = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(n);
    for (Tind i = 0; i < n; ++i) xr[i] = br[i];
    // Forward solve L z = b: propagate each eliminated vertex to its parent.
    for (std::size_t e = 0; e < ne; ++e) xr[par[e]] -= mult[e] * xr[ord[e]];
    // Diagonal scale by D^{-1}.
    for (Tind i = 0; i < n; ++i) xr[i] *= dinv[i];
    // Backward solve L^T x = z: reverse elimination order.
    for (std::size_t e = ne; e-- > 0;) xr[ord[e]] -= mult[e] * xr[par[e]];
  }
}

/// A single fixed-pattern forest matrix: detect + factorize, update, solve.
template <typename Tind, typename Tval>
class ForestLDL {
 public:
  /// Build the elimination structure, detect the forest, and (if so) factorize.
  ForestLDL(const Tind* indptr, const Tind* indices, const Tval* data, Tind n) {
    struct_.build(indptr, indices, n);
    if (struct_.is_forest) {
      mult_.assign(struct_.order.size(), Tval(0));
      dinv_.assign(static_cast<std::size_t>(struct_.n), Tval(0));
      forest_factor_into(struct_, data, mult_.data(), dinv_.data());
    }
  }

  /// Whether the off-diagonal graph is acyclic (a forest).
  bool is_forest() const { return struct_.is_forest; }

  /// System dimension.
  Tind n() const { return struct_.n; }

  /// The pattern-only elimination plan (shared with a batch over this pattern).
  const ForestStructure<Tind>& structure() const { return struct_; }

  /// Numeric refactorization with new values on the same (forest) pattern.
  void update(const Tval* data) {
    if (!struct_.is_forest) {
      throw std::runtime_error("ForestLDL::update called on a non-forest matrix");
    }
    forest_factor_into(struct_, data, mult_.data(), dinv_.data());
  }

  /// Solve M X = B for k right-hand sides ([k, n] row-major), parallel over RHS.
  void solve(const Tval* b, Tval* x, Tind k) const {
    if (!struct_.is_forest) {
      throw std::runtime_error("ForestLDL::solve called on a non-forest matrix");
    }
    const ForestStructure<Tind>& s = struct_;
    const Tval* mult = mult_.data();
    const Tval* dinv = dinv_.data();
    const std::size_t n = static_cast<std::size_t>(struct_.n);
    parallel_for(static_cast<std::size_t>(k), [&](std::size_t r) {
      forest_solve_with(s, mult, dinv, b + r * n, x + r * n, Tind(1));
    });
  }

 private:
  ForestStructure<Tind> struct_;
  std::vector<Tval> mult_;  // L multipliers (refactored each update)
  std::vector<Tval> dinv_;  // 1 / pivot per vertex
};

/**
 * Batched exact forest factorization+solve over B matrices sharing one pattern.
 *
 * The elimination structure is built once; each system's numeric factor is
 * independent, so a single GIL-released call factors+solves all B systems in
 * parallel over the std::thread pool.  Per-system factor outputs live in arenas
 * sized to the largest B seen and reused across calls (no per-system malloc in
 * the hot loop).
 */
template <typename Tind, typename Tval>
class ForestBatchSolver {
 public:
  ForestBatchSolver(const Tind* indptr, const Tind* indices, Tind n) {
    struct_.build(indptr, indices, n);
  }

  bool is_forest() const { return struct_.is_forest; }
  Tind n() const { return struct_.n; }
  std::size_t nnz() const { return struct_.nnz; }

  /**
   * Per-system batch: values [B, nnz] row-major (system s = values + s*nnz); rhs
   * is a flat [totalK, n] row-major buffer with rhs_offsets[B+1] (system s owns
   * rows [rhs_offsets[s], rhs_offsets[s+1])); x_out matches rhs.
   */
  void solve_batch(const Tval* values, std::size_t batch, const Tval* rhs,
                   const int64_t* rhs_offsets, Tval* x_out) {
    require_forest();
    const std::size_t ne = struct_.order.size();
    const std::size_t n = static_cast<std::size_t>(struct_.n);
    const std::size_t nnz = struct_.nnz;
    ensure_arena(batch);
    const ForestStructure<Tind>& s = struct_;
    Tval* mult_arena = mult_arena_.data();
    Tval* dinv_arena = dinv_arena_.data();
    parallel_for(batch, [&](std::size_t sys) {
      Tval* mult = mult_arena + sys * ne;
      Tval* dinv = dinv_arena + sys * n;
      forest_factor_into(s, values + sys * nnz, mult, dinv);
      const int64_t off = rhs_offsets[sys];
      const Tind k = static_cast<Tind>(rhs_offsets[sys + 1] - off);
      forest_solve_with(s, mult, dinv, rhs + static_cast<std::size_t>(off) * n,
                        x_out + static_cast<std::size_t>(off) * n, k);
    });
  }

  /// Shared matrix, B RHS: values [nnz]; rhs and x_out are [B, n] row-major.
  void solve_batch_shared(const Tval* values, std::size_t batch, const Tval* rhs,
                          Tval* x_out) {
    require_forest();
    const std::size_t n = static_cast<std::size_t>(struct_.n);
    ensure_arena(1);
    Tval* mult = mult_arena_.data();
    Tval* dinv = dinv_arena_.data();
    forest_factor_into(struct_, values, mult, dinv);
    const ForestStructure<Tind>& s = struct_;
    parallel_for(batch, [&](std::size_t sys) {
      forest_solve_with(s, mult, dinv, rhs + sys * n, x_out + sys * n, Tind(1));
    });
  }

 private:
  void require_forest() const {
    if (!struct_.is_forest) {
      throw std::runtime_error("ForestBatchSolver: matrix is not a forest");
    }
  }

  void ensure_arena(std::size_t batch) {
    const std::size_t ne = struct_.order.size();
    const std::size_t n = static_cast<std::size_t>(struct_.n);
    if (mult_arena_.size() < batch * ne) mult_arena_.resize(batch * ne);
    if (dinv_arena_.size() < batch * n) dinv_arena_.resize(batch * n);
  }

  ForestStructure<Tind> struct_;
  std::vector<Tval> mult_arena_;  // [B, ne] high-water-mark, reused across calls
  std::vector<Tval> dinv_arena_;  // [B, n]
};

}  // namespace lsolve
