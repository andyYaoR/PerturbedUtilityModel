// Direct sparse Cholesky (CHOLMOD) for SPD systems - the fast path for the PURC
// matrix M = H + eps*I (SPD) and, more generally, for the low-fill regime where
// direct factorization beats iterative solvers (road networks at any practical
// size).  This is the same library Julia's `cholesky` uses.
//
// The matrix pattern is fixed across a PURC run, so the symbolic analysis
// (fill-reducing ordering) is computed ONCE; weight/regularizer changes trigger
// a cheap numeric-only refactorization.  Single and batched right-hand sides
// reuse the factor; solve2 reuses workspace to minimise per-call overhead.
#pragma once

#include <cholmod.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lsolve {

// Owns a CHOLMOD common + factor for one fixed-pattern SPD matrix.
class CholmodSolver {
 public:
  // Factorize a symmetric SPD matrix given as a full (both-triangles) sorted
  // CSC.  Uses the lower triangle (stype = -1); analyze + numeric factorize.
  CholmodSolver(const int64_t* indptr, const int64_t* indices, const double* data,
                std::size_t n, std::size_t nnz)
      : n_(n), p_(indptr, indptr + n + 1), i_(indices, indices + nnz),
        x_(data, data + nnz) {
    cholmod_l_start(&c_);
    c_.final_ll = 1;        // produce an LL^T factor
    c_.print = 0;           // silent
    set_view();
    factor_ = cholmod_l_analyze(&a_, &c_);  // symbolic: fill-reducing ordering
    if (factor_ == nullptr) {
      cleanup();
      throw std::runtime_error("CHOLMOD analyze failed");
    }
    refactorize();
  }

  CholmodSolver(const CholmodSolver&) = delete;
  CholmodSolver& operator=(const CholmodSolver&) = delete;

  ~CholmodSolver() { cleanup(); }

  std::size_t n() const { return n_; }

  // Numeric refactorization with new values on the SAME pattern (reuses the
  // symbolic analysis): the "update weights / eps without re-analyzing" path.
  void update(const double* data, std::size_t nnz) {
    if (nnz != x_.size()) {
      throw std::runtime_error("CholmodSolver.update: nnz changed (pattern differs)");
    }
    std::copy(data, data + nnz, x_.begin());
    a_.x = x_.data();
    refactorize();
  }

  // Solve M x = b for k right-hand sides (b and x_out are [n, k] column-major,
  // i.e. each system contiguous - matching a NumPy [k, n] row-major buffer).
  void solve(const double* b, double* x_out, std::size_t k) {
    cholmod_dense rhs;
    rhs.nrow = n_;
    rhs.ncol = k;
    rhs.nzmax = n_ * k;
    rhs.d = n_;
    rhs.x = const_cast<double*>(b);
    rhs.z = nullptr;
    rhs.xtype = CHOLMOD_REAL;
    rhs.dtype = CHOLMOD_DOUBLE;
    // solve2 reuses x_work_/y_work_/e_work_ across calls (no per-call alloc).
    if (!cholmod_l_solve2(CHOLMOD_A, factor_, &rhs, nullptr, &x_work_, nullptr,
                          &y_work_, &e_work_, &c_)) {
      throw std::runtime_error("CHOLMOD solve failed");
    }
    std::copy(static_cast<double*>(x_work_->x),
              static_cast<double*>(x_work_->x) + n_ * k, x_out);
  }

 private:
  void set_view() {
    a_.nrow = n_;
    a_.ncol = n_;
    a_.nzmax = x_.size();
    a_.p = p_.data();
    a_.i = i_.data();
    a_.nz = nullptr;
    a_.x = x_.data();
    a_.z = nullptr;
    a_.stype = -1;  // symmetric; use the lower triangle of the full CSC
    a_.itype = CHOLMOD_LONG;
    a_.xtype = CHOLMOD_REAL;
    a_.dtype = CHOLMOD_DOUBLE;
    a_.sorted = 1;
    a_.packed = 1;
  }

  void refactorize() {
    if (!cholmod_l_factorize(&a_, factor_, &c_) || c_.status != CHOLMOD_OK) {
      throw std::runtime_error(
          "CHOLMOD factorize failed (matrix not positive definite?)");
    }
  }

  void cleanup() {
    if (x_work_) cholmod_l_free_dense(&x_work_, &c_);
    if (y_work_) cholmod_l_free_dense(&y_work_, &c_);
    if (e_work_) cholmod_l_free_dense(&e_work_, &c_);
    if (factor_) cholmod_l_free_factor(&factor_, &c_);
    cholmod_l_finish(&c_);
  }

  std::size_t n_;
  std::vector<int64_t> p_;  // owned CSC (pattern fixed, values updatable)
  std::vector<int64_t> i_;
  std::vector<double> x_;
  cholmod_common c_;
  cholmod_sparse a_{};
  cholmod_factor* factor_ = nullptr;
  cholmod_dense* x_work_ = nullptr;  // persistent solve2 workspace
  cholmod_dense* y_work_ = nullptr;
  cholmod_dense* e_work_ = nullptr;
};

}  // namespace lsolve
