/**
 * nanobind bindings for the LaplacianSolve native core.
 *
 * Exposes the CPU solve path (Stage 1b): the LDLinv triangular solve and the
 * approxChol-preconditioned PCG against a Laplacian.  All arrays are borrowed
 * zero-copy (numpy/scipy/torch-cpu) and the GIL is released around compute,
 * mirroring PUM's nb_spmv_module.cpp.  The approxChol build, SDDM wrapper, and
 * forest solve are added in later sub-milestones.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "approx_chol_build.hpp"
#include "forest_ldl.hpp"
#include "ldl_solve.hpp"
#include "parallel.hpp"
#include "pcg.hpp"
#include "spmv.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// 1-D contiguous CPU array views (zero-copy).
using Int64Arr1D =
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Float64Arr1D =
    nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Float64MutArr1D =
    nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Int64MutArr1D =
    nb::ndarray<nb::numpy, int64_t, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Float64Arr2D =
    nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::device::cpu, nb::c_contig>;
using Float64MutArr2D =
    nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::device::cpu, nb::c_contig>;

namespace {

// In-place y_out += a * x; validates the zero-copy ndarray + GIL-release path.
void axpy_f64(double a, Float64Arr1D x, Float64MutArr1D y) {
  const std::size_t n = x.shape(0);
  const double* xp = x.data();
  double* yp = y.data();
  nb::gil_scoped_release release;
  for (std::size_t i = 0; i < n; ++i) {
    yp[i] += a * xp[i];
  }
}

// Batch thread-pool control.  num_threads caps the across-system parallelism of
// the native batch solves (one thread per independent system / RHS).
int max_threads() { return lsolve::ThreadPool::instance().num_threads(); }

void set_threads(int n) { lsolve::ThreadPool::instance().set_num_threads(n); }

// x_out = (L D L^T)^{-1} b for the approxChol LDLinv given by its arrays.
void ldl_solve_f64(Int64Arr1D col, Int64Arr1D colptr, Int64Arr1D rowval,
                   Float64Arr1D fval, Float64Arr1D d, Float64Arr1D b,
                   Float64MutArr1D x_out, bool subtract_mean) {
  const std::size_t ncol = col.shape(0);
  const std::size_t n = d.shape(0);
  const int64_t* pcol = col.data();
  const int64_t* pcolptr = colptr.data();
  const int64_t* prowval = rowval.data();
  const double* pfval = fval.data();
  const double* pd = d.data();
  const double* pb = b.data();
  double* px = x_out.data();
  nb::gil_scoped_release release;
  lsolve::ldl_solve<int64_t, double>(pcol, pcolptr, prowval, pfval, pd, ncol, n,
                                     pb, px, subtract_mean);
}

// PCG for a Laplacian (CSC) preconditioned by the approxChol LDLinv.
// x_out is the warm start (zeros for a cold start) and receives the solution.
// Returns (iterations, relative_residual, converged).
std::tuple<int, double, bool> pcg_lap_f64(
    Int64Arr1D indptr, Int64Arr1D indices, Float64Arr1D data, Float64Arr1D b,
    Int64Arr1D col, Int64Arr1D colptr, Int64Arr1D rowval, Float64Arr1D fval,
    Float64Arr1D d, Float64MutArr1D x_out, double tol, int maxits,
    int stag_test) {
  const std::size_t n = b.shape(0);
  const std::size_t ncol = col.shape(0);
  const int64_t* ip = indptr.data();
  const int64_t* ic = indices.data();
  const double* da = data.data();
  const double* pb = b.data();
  const int64_t* pcol = col.data();
  const int64_t* pcolptr = colptr.data();
  const int64_t* prowval = rowval.data();
  const double* pfval = fval.data();
  const double* pd = d.data();
  double* px = x_out.data();

  nb::gil_scoped_release release;
  // The Laplacian is symmetric, so its CSC arrays are also valid CSR; use the
  // race-free row-walk kernel (OpenMP-parallel for large n).
  auto matvec = [&](const double* in, double* out) {
    lsolve::csr_matvec<int64_t, double>(ip, ic, da, n, in, out);
  };
  auto precond = [&](const double* in, double* out) {
    lsolve::ldl_solve<int64_t, double>(pcol, pcolptr, prowval, pfval, pd, ncol,
                                       n, in, out, /*subtract_mean=*/true);
  };
  auto res = lsolve::pcg<double>(n, pb, matvec, precond, tol, maxits, stag_test,
                                 px);
  return std::make_tuple(res.iterations, res.relres, res.converged);
}

// Native batched PCG: solve B right-hand sides (rows of b) against ONE shared
// Laplacian + LDLinv, in parallel via OpenMP over the batch.  One GIL release
// covers the whole batch (no per-solve Python overhead); nested OpenMP is off
// by default so each per-system solve runs serially.  x_out warm-starts and
// receives the solutions; iters_out/conv_out get per-system diagnostics.
void pcg_lap_batch_f64(Int64Arr1D indptr, Int64Arr1D indices, Float64Arr1D data,
                       Float64Arr2D b, Int64Arr1D col, Int64Arr1D colptr,
                       Int64Arr1D rowval, Float64Arr1D fval, Float64Arr1D d,
                       Float64MutArr2D x_out, Int64MutArr1D iters_out,
                       Int64MutArr1D conv_out, double tol, int maxits,
                       int stag_test) {
  const std::size_t batch = b.shape(0);
  const std::size_t n = b.shape(1);
  const std::size_t ncol = col.shape(0);
  const int64_t* ip = indptr.data();
  const int64_t* ic = indices.data();
  const double* da = data.data();
  const int64_t* pcol = col.data();
  const int64_t* pcolptr = colptr.data();
  const int64_t* prowval = rowval.data();
  const double* pfval = fval.data();
  const double* pd = d.data();
  const double* bptr = b.data();
  double* xptr = x_out.data();
  int64_t* its = iters_out.data();
  int64_t* cvs = conv_out.data();

  nb::gil_scoped_release release;
  lsolve::parallel_for(batch, [&](std::size_t s) {
    const double* bs = bptr + s * n;
    double* xs = xptr + s * n;
    auto matvec = [&](const double* in, double* out) {
      lsolve::csr_matvec<int64_t, double>(ip, ic, da, n, in, out);
    };
    auto precond = [&](const double* in, double* out) {
      lsolve::ldl_solve<int64_t, double>(pcol, pcolptr, prowval, pfval, pd, ncol, n,
                                         in, out, /*subtract_mean=*/true);
    };
    auto res = lsolve::pcg<double>(n, bs, matvec, precond, tol, maxits, stag_test, xs);
    its[s] = res.iterations;
    cvs[s] = res.converged ? 1 : 0;
  });
}

// Native batched LDLinv apply: x_out[s] = (L D L^T)^{-1} b[s] for each row, in
// parallel over the batch (the exact forest fast path for many RHS).
void ldl_solve_batch_f64(Int64Arr1D col, Int64Arr1D colptr, Int64Arr1D rowval,
                         Float64Arr1D fval, Float64Arr1D d, Float64Arr2D b,
                         Float64MutArr2D x_out, bool subtract_mean) {
  const std::size_t batch = b.shape(0);
  const std::size_t n = b.shape(1);
  const std::size_t ncol = col.shape(0);
  const int64_t* pcol = col.data();
  const int64_t* pcolptr = colptr.data();
  const int64_t* prowval = rowval.data();
  const double* pfval = fval.data();
  const double* pd = d.data();
  const double* bptr = b.data();
  double* xptr = x_out.data();

  nb::gil_scoped_release release;
  lsolve::parallel_for(batch, [&](std::size_t s) {
    lsolve::ldl_solve<int64_t, double>(pcol, pcolptr, prowval, pfval, pd, ncol, n,
                                       bptr + s * n, xptr + s * n, subtract_mean);
  });
}

// Assemble the SDDM matrix values M = C diag(w) C^T + eps*I for a batch of
// systems, in the cached CSC order, parallel over systems.  A pure scatter:
// every off-diagonal CSC slot of edge e gets -w_e, and every diagonal slot gets
// eps plus the sum of its incident edge weights.  The CSC-position maps are
// built once on the Python side (edge -> the two off-diagonal slots and the two
// endpoint-diagonal slots; node -> its diagonal slot).  Inactive edges are
// simply passed with w_e = 0 (the caller folds any active mask into w).
void assemble_sddm_values_f64(Int64Arr1D diag_pos, Int64Arr1D off0, Int64Arr1D off1,
                              Int64Arr1D ediag0, Int64Arr1D ediag1, Float64Arr2D weights,
                              Float64Arr1D eps, Float64MutArr2D values_out) {
  const std::size_t batch = values_out.shape(0);
  const std::size_t nnz = values_out.shape(1);
  const std::size_t n = diag_pos.shape(0);
  const std::size_t m = off0.shape(0);
  const int64_t* dp = diag_pos.data();
  const int64_t* o0 = off0.data();
  const int64_t* o1 = off1.data();
  const int64_t* d0 = ediag0.data();
  const int64_t* d1 = ediag1.data();
  const double* w = weights.data();
  const double* ep = eps.data();
  double* out = values_out.data();

  nb::gil_scoped_release release;
  lsolve::parallel_for(batch, [&](std::size_t s) {
    double* vals = out + s * nnz;
    const double eps_s = ep[s];
    for (std::size_t i = 0; i < n; ++i) vals[dp[i]] = eps_s;  // diagonal seeded with eps
    const double* ws = w + s * m;
    for (std::size_t e = 0; e < m; ++e) {
      const double we = ws[e];
      vals[o0[e]] = -we;   // M[u, v]
      vals[o1[e]] = -we;   // M[v, u]
      vals[d0[e]] += we;   // M[u, u] += w_e
      vals[d1[e]] += we;   // M[v, v] += w_e
    }
  });
}

// Wrap a std::vector as an owning 1-D NumPy array by moving it into a
// capsule-managed heap buffer (no element copy).  Call with the GIL held.
template <typename T>
nb::ndarray<nb::numpy, T> to_numpy_1d(std::vector<T>&& v) {
  auto* held = new std::vector<T>(std::move(v));
  nb::capsule owner(held,
                    [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  const std::size_t n = held->size();
  return nb::ndarray<nb::numpy, T>(held->data(), {n}, owner);
}

using LdlTuple =
    std::tuple<nb::ndarray<nb::numpy, int64_t>, nb::ndarray<nb::numpy, int64_t>,
               nb::ndarray<nb::numpy, int64_t>, nb::ndarray<nb::numpy, double>,
               nb::ndarray<nb::numpy, double>>;

// Build the approxChol LDLinv (:deg order) from a CSC adjacency + sample stream.
// Returns (col, colptr, rowval, fval, d).
LdlTuple approx_chol_f64(Int64Arr1D indptr, Int64Arr1D indices, Float64Arr1D data,
                         Float64Arr1D samples) {
  const std::size_t n = indptr.shape(0) - 1;
  const int64_t* ip = indptr.data();
  const int64_t* ic = indices.data();
  const double* da = data.data();
  const double* sp = samples.data();
  const std::size_t ns = samples.shape(0);

  lsolve::LdlFactor<int64_t, double> f;
  {
    nb::gil_scoped_release release;
    f = lsolve::approx_chol<int64_t, double>(ip, ic, da, n, sp, ns);
  }
  return std::make_tuple(to_numpy_1d(std::move(f.col)), to_numpy_1d(std::move(f.colptr)),
                         to_numpy_1d(std::move(f.rowval)), to_numpy_1d(std::move(f.fval)),
                         to_numpy_1d(std::move(f.d)));
}

// Build the approxChol LDLinv using a seeded internal RNG (the production path,
// no Python sample array needed).  Returns (col, colptr, rowval, fval, d).
LdlTuple approx_chol_seeded_f64(Int64Arr1D indptr, Int64Arr1D indices,
                                Float64Arr1D data, uint64_t seed) {
  const std::size_t n = indptr.shape(0) - 1;
  const int64_t* ip = indptr.data();
  const int64_t* ic = indices.data();
  const double* da = data.data();

  lsolve::LdlFactor<int64_t, double> f;
  {
    nb::gil_scoped_release release;
    f = lsolve::approx_chol_seeded<int64_t, double>(ip, ic, da, n, seed);
  }
  return std::make_tuple(to_numpy_1d(std::move(f.col)), to_numpy_1d(std::move(f.colptr)),
                         to_numpy_1d(std::move(f.rowval)), to_numpy_1d(std::move(f.fval)),
                         to_numpy_1d(std::move(f.d)));
}

}  // namespace

NB_MODULE(_laplaciansolve_core, m) {
  m.doc() = "LaplacianSolve native core (approxChol port of Laplacians.jl).";
  m.attr("__core_version__") = "0.0.1";

  m.def("axpy_f64", &axpy_f64, "a"_a, "x"_a, "y_out"_a,
        "In-place y_out += a * x (float64); zero-copy ndarray smoke test.");

  m.def("max_threads", &max_threads,
        "Thread-pool parallelism cap for native batch solves (defaults to hardware).");
  m.def("set_threads", &set_threads, "n"_a,
        "Set the thread-pool parallelism cap for native batch solves (clamped to hardware).");

  m.def("ldl_solve_f64", &ldl_solve_f64, "col"_a, "colptr"_a, "rowval"_a,
        "fval"_a, "d"_a, "b"_a, "x_out"_a, "subtract_mean"_a,
        "Apply the approxChol LDLinv: x_out = (L D L^T)^{-1} b (float64/int64).");

  m.def("pcg_lap_f64", &pcg_lap_f64, "indptr"_a, "indices"_a, "data"_a, "b"_a,
        "col"_a, "colptr"_a, "rowval"_a, "fval"_a, "d"_a, "x_out"_a, "tol"_a,
        "maxits"_a, "stag_test"_a,
        "approxChol-preconditioned PCG for a Laplacian (CSC). x_out warm-starts "
        "and receives the solution. Returns (iterations, relres, converged).");

  m.def("pcg_lap_batch_f64", &pcg_lap_batch_f64, "indptr"_a, "indices"_a, "data"_a,
        "b"_a, "col"_a, "colptr"_a, "rowval"_a, "fval"_a, "d"_a, "x_out"_a,
        "iters_out"_a, "conv_out"_a, "tol"_a, "maxits"_a, "stag_test"_a,
        "Native batched approxChol-PCG: solve the rows of b against one shared "
        "Laplacian + LDLinv in parallel (OpenMP over the batch).");

  m.def("ldl_solve_batch_f64", &ldl_solve_batch_f64, "col"_a, "colptr"_a, "rowval"_a,
        "fval"_a, "d"_a, "b"_a, "x_out"_a, "subtract_mean"_a,
        "Native batched LDLinv apply over the rows of b (OpenMP); exact forest path.");

  m.def("assemble_sddm_values_f64", &assemble_sddm_values_f64, "diag_pos"_a, "off0"_a, "off1"_a,
        "ediag0"_a, "ediag1"_a, "weights"_a, "eps"_a, "values_out"_a,
        "Assemble M = C diag(w) C^T + eps*I values [B, nnz] from edge weights [B, m] in the "
        "cached CSC order, parallel over systems (the PURC per-iteration value refill).");

  m.def("approx_chol_f64", &approx_chol_f64, "indptr"_a, "indices"_a, "data"_a,
        "samples"_a,
        "Build the approxChol LDLinv (:deg order) from a CSC adjacency and a "
        "uniform [0,1) sample stream. Returns (col, colptr, rowval, fval, d).");

  m.def("approx_chol_seeded_f64", &approx_chol_seeded_f64, "indptr"_a, "indices"_a,
        "data"_a, "seed"_a,
        "Build the approxChol LDLinv (:deg order) from a CSC adjacency using a "
        "seeded internal RNG. Returns (col, colptr, rowval, fval, d).");

  // Exact zero-fill LDL for SDDM matrices whose off-diagonal graph is a forest
  // (the acyclic fast path; the constructor also *detects* the forest cheaply).
  using ForestSolver = lsolve::ForestLDL<int64_t, double>;
  nb::class_<ForestSolver>(m, "ForestSolver")
      .def(
          "__init__",
          [](ForestSolver* self, Int64Arr1D indptr, Int64Arr1D indices, Float64Arr1D data,
             int64_t n) {
            nb::gil_scoped_release release;
            new (self)
                ForestSolver(indptr.data(), indices.data(), data.data(), static_cast<int64_t>(n));
          },
          "indptr"_a, "indices"_a, "data"_a, "n"_a,
          "Leaf-prune to detect a forest; if acyclic, build the exact zero-fill LDL "
          "(full sorted CSC, int64).")
      .def("is_forest", &ForestSolver::is_forest,
           "Whether the off-diagonal graph is acyclic (else route to CHOLMOD).")
      .def(
          "update",
          [](ForestSolver& self, Float64Arr1D data) {
            const double* p = data.data();
            nb::gil_scoped_release release;
            self.update(p);
          },
          "data"_a, "Numeric refactorization with new values on the same forest pattern.")
      .def(
          "solve",
          [](ForestSolver& self, Float64Arr1D b, Float64MutArr1D x_out, int64_t k) {
            const double* pb = b.data();
            double* px = x_out.data();
            nb::gil_scoped_release release;
            self.solve(pb, px, static_cast<int64_t>(k));
          },
          "b"_a, "x_out"_a, "k"_a,
          "Solve M x = b for k RHS (flat n*k buffers, [k, n] row-major).");

  // Batched exact forest factorization+solve over B matrices sharing one
  // pattern (the PURC per-destination case): the elimination plan is built once,
  // then B numeric factors + solves run in one GIL-released call, parallel over
  // systems.  See ForestBatchSolver in forest_ldl.hpp.
  using ForestBatch = lsolve::ForestBatchSolver<int64_t, double>;
  nb::class_<ForestBatch>(m, "ForestBatchSolver")
      .def(
          "__init__",
          [](ForestBatch* self, Int64Arr1D indptr, Int64Arr1D indices, int64_t n) {
            nb::gil_scoped_release release;
            new (self) ForestBatch(indptr.data(), indices.data(), static_cast<int64_t>(n));
          },
          "indptr"_a, "indices"_a, "n"_a,
          "Leaf-prune the shared pattern to detect a forest and build the elimination plan.")
      .def("is_forest", &ForestBatch::is_forest,
           "Whether the shared off-diagonal graph is acyclic (else route to CHOLMOD).")
      .def("n", [](ForestBatch& self) { return static_cast<int64_t>(self.n()); }, "Vertex count.")
      .def("nnz", [](ForestBatch& self) { return static_cast<int64_t>(self.nnz()); },
           "Stored nonzeros of the shared CSC pattern.")
      .def(
          "solve_batch",
          [](ForestBatch& self, Float64Arr2D values, Float64Arr2D rhs, Int64Arr1D rhs_offsets,
             Float64MutArr2D x_out) {
            const std::size_t batch = values.shape(0);
            const double* pv = values.data();
            const double* pr = rhs.data();
            const int64_t* po = rhs_offsets.data();
            double* px = x_out.data();
            nb::gil_scoped_release release;
            self.solve_batch(pv, batch, pr, po, px);
          },
          "values"_a, "rhs"_a, "rhs_offsets"_a, "x_out"_a,
          "Per-system batch: values [B, nnz]; rhs/x_out flat [totalK, n] row-major with "
          "int64 rhs_offsets[B+1] (system s owns rows [offsets[s], offsets[s+1])).")
      .def(
          "solve_batch_shared",
          [](ForestBatch& self, Float64Arr1D values, Float64Arr2D rhs, Float64MutArr2D x_out) {
            const std::size_t batch = rhs.shape(0);
            const double* pv = values.data();
            const double* pr = rhs.data();
            double* px = x_out.data();
            nb::gil_scoped_release release;
            self.solve_batch_shared(pv, batch, pr, px);
          },
          "values"_a, "rhs"_a, "x_out"_a,
          "Shared matrix, B RHS: values [nnz]; rhs/x_out [B, n] row-major.");
}
