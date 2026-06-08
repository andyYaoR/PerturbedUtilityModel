// Batched direct sparse Cholesky (CHOLMOD) over B SPD matrices that share one
// fixed sparsity pattern - the PURC per-destination case, where each system has
// its own weights (hence its own M = H + eps*I) on the common network pattern.
//
// CHOLMOD has no native batch-factorize (cholmod_factorize is one matrix -> one
// factor), so we parallelize the B factorizations across the std::thread pool.
// A cholmod_common is not thread-safe, so each pool worker owns a private common
// + a factor cloned (symbolically) from a single shared analysis, plus its own
// solve2 workspaces.  parallel_for_ranges hands each worker a stable id in
// [0, P) for the duration of a call, so workers[id] is touched by exactly one
// thread - no locking.  The analysis (fill-reducing ordering) is computed once
// and shared by all clones (cholmod_l_copy_factor); per-system multi-RHS solves
// use CHOLMOD's native dense ncol path.
//
// CHOLMOD's *own* OpenMP is capped at 1 thread per worker common (nthreads_max),
// because parallelism is across systems here; intra-op OpenMP would only
// oversubscribe.  Single-matrix solves (CholmodSolver) keep full CHOLMOD OpenMP.
#pragma once

#include <cholmod.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"

namespace lsolve {

// Per-system factorization status written by factorize_and_solve_batch.
enum CholmodBatchStatus : int64_t {
  kCholmodOk = 0,
  kCholmodNotPosDef = 1,  // cholmod_factorize failed (matrix not SPD)
  kCholmodSolveFailed = 2,
};

class BatchedCholmodSolver {
 public:
  // Analyze a symmetric SPD pattern given as a full sorted CSC (int64).  The
  // values of the first matrix are only used to validate the analysis; per-call
  // values arrive in factorize_and_solve_batch.
  BatchedCholmodSolver(const int64_t* indptr, const int64_t* indices, const double* data,
                       std::size_t n, std::size_t nnz)
      : n_(n), nnz_(nnz), p_(indptr, indptr + n + 1), i_(indices, indices + nnz) {
    cholmod_l_start(&master_);
    configure_common(master_);
    x_master_.assign(data, data + nnz);  // a valid (unused-by-analyze) value view
    fill_sparse(master_view_, x_master_.data());
    symbolic_ = cholmod_l_analyze(&master_view_, &master_);
    if (symbolic_ == nullptr) {
      cholmod_l_finish(&master_);
      throw std::runtime_error("CHOLMOD analyze failed");
    }
    // Size workers to the hardware maximum so any later thread-cap (always
    // <= hardware) yields worker ids within range - no per-call clamp needed.
    P_ = ThreadPool::hardware();
    workers_.resize(static_cast<std::size_t>(P_));
    for (int w = 0; w < P_; ++w) {
      Worker& worker = workers_[static_cast<std::size_t>(w)];
      cholmod_l_start(&worker.c);
      configure_common(worker.c);
      worker.started = true;
      worker.values.assign(nnz_, 0.0);
      fill_sparse(worker.a, worker.values.data());
      worker.factor = cholmod_l_copy_factor(symbolic_, &worker.c);
      if (worker.factor == nullptr) {
        cleanup();
        throw std::runtime_error("CHOLMOD copy_factor failed");
      }
    }
  }

  BatchedCholmodSolver(const BatchedCholmodSolver&) = delete;
  BatchedCholmodSolver& operator=(const BatchedCholmodSolver&) = delete;

  ~BatchedCholmodSolver() { cleanup(); }

  std::size_t n() const { return n_; }
  std::size_t nnz() const { return nnz_; }

  // Factorize + solve B systems in one GIL-released call, parallel over systems.
  //   values:      [B, nnz] row-major (system s = values + s*nnz)
  //   rhs:         flat [totalK, n] row-major
  //   rhs_offsets: [B+1] int64; system s owns rows [offsets[s], offsets[s+1])
  //   x_out:       flat [totalK, n] row-major (matches rhs)
  //   status_out:  [B] int64; per-system CholmodBatchStatus (0 = ok)
  void factorize_and_solve_batch(const double* values, std::size_t batch, const double* rhs,
                                 const int64_t* rhs_offsets, double* x_out, int64_t* status_out) {
    const std::size_t n = n_;
    const std::size_t nnz = nnz_;
    parallel_for_ranges(batch, [&](std::size_t begin, std::size_t end, int w) {
      Worker& worker = workers_[static_cast<std::size_t>(w)];
      for (std::size_t s = begin; s < end; ++s) {
        const int64_t off = rhs_offsets[s];
        const int64_t k = rhs_offsets[s + 1] - off;
        if (k <= 0) {  // empty RHS block (ragged): nothing to factor/solve
          status_out[s] = kCholmodOk;
          continue;
        }
        std::copy(values + s * nnz, values + (s + 1) * nnz, worker.values.begin());
        worker.a.x = worker.values.data();
        if (!cholmod_l_factorize(&worker.a, worker.factor, &worker.c) ||
            worker.c.status != CHOLMOD_OK) {
          status_out[s] = kCholmodNotPosDef;
          continue;
        }
        cholmod_dense b{};
        b.nrow = n;
        b.ncol = static_cast<std::size_t>(k);
        b.nzmax = n * static_cast<std::size_t>(k);
        b.d = n;
        b.x = const_cast<double*>(rhs + static_cast<std::size_t>(off) * n);
        b.z = nullptr;
        b.xtype = CHOLMOD_REAL;
        b.dtype = CHOLMOD_DOUBLE;
        if (!cholmod_l_solve2(CHOLMOD_A, worker.factor, &b, nullptr, &worker.x, nullptr,
                              &worker.y, &worker.e, &worker.c)) {
          status_out[s] = kCholmodSolveFailed;
          continue;
        }
        const double* xs = static_cast<double*>(worker.x->x);
        std::copy(xs, xs + n * static_cast<std::size_t>(k),
                  x_out + static_cast<std::size_t>(off) * n);
        status_out[s] = kCholmodOk;
      }
    });
  }

 private:
  struct Worker {
    cholmod_common c{};
    cholmod_sparse a{};
    cholmod_factor* factor = nullptr;
    cholmod_dense* x = nullptr;  // persistent solve2 workspaces (per worker)
    cholmod_dense* y = nullptr;
    cholmod_dense* e = nullptr;
    std::vector<double> values;  // this worker's copy of the active system's values
    bool started = false;        // whether cholmod_l_start(&c) has run (for cleanup)
  };

  static void configure_common(cholmod_common& c) {
    c.final_ll = 1;        // LL^T factor (matches CholmodSolver)
    c.print = 0;           // silent
    c.nthreads_max = 1;    // across-system parallelism is ours; no nested OpenMP
    c.chunk = 1e18;        // never split a factorization across OpenMP threads
  }

  void fill_sparse(cholmod_sparse& a, double* xvals) {
    a.nrow = n_;
    a.ncol = n_;
    a.nzmax = nnz_;
    a.p = p_.data();
    a.i = i_.data();
    a.nz = nullptr;
    a.x = xvals;
    a.z = nullptr;
    a.stype = -1;  // symmetric; lower triangle of the full sorted CSC
    a.itype = CHOLMOD_LONG;
    a.xtype = CHOLMOD_REAL;
    a.dtype = CHOLMOD_DOUBLE;
    a.sorted = 1;
    a.packed = 1;
  }

  void cleanup() {
    for (Worker& worker : workers_) {
      if (!worker.started) continue;
      if (worker.x) cholmod_l_free_dense(&worker.x, &worker.c);
      if (worker.y) cholmod_l_free_dense(&worker.y, &worker.c);
      if (worker.e) cholmod_l_free_dense(&worker.e, &worker.c);
      if (worker.factor) cholmod_l_free_factor(&worker.factor, &worker.c);
      cholmod_l_finish(&worker.c);
    }
    workers_.clear();
    if (symbolic_) cholmod_l_free_factor(&symbolic_, &master_);
    cholmod_l_finish(&master_);
  }

  std::size_t n_;
  std::size_t nnz_;
  std::vector<int64_t> p_;  // shared CSC pattern (read-only across workers)
  std::vector<int64_t> i_;
  std::vector<double> x_master_;
  cholmod_common master_;
  cholmod_sparse master_view_{};
  cholmod_factor* symbolic_ = nullptr;  // shared symbolic analysis (clone source)
  int P_ = 1;
  std::vector<Worker> workers_;
};

}  // namespace lsolve
