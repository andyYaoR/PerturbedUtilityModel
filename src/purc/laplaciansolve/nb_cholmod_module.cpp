/**
 * nanobind bindings for the CHOLMOD direct SPD solver (_laplaciansolve_cholmod).
 *
 * Built only when SuiteSparse/CHOLMOD is found at configure time.  Exposes a
 * CholmodSolver object: factorize once (in the constructor), then reuse the
 * factor for many single / batched right-hand sides and for numeric-only
 * refactorizations (update).  All heavy calls release the GIL.
 *
 * Buffer convention: a batch of k right-hand sides is a flat float64 array of
 * length n*k laid out as a NumPy [k, n] row-major array (== [n, k] column-major,
 * which is what CHOLMOD's dense format expects), so reshape((k, n)) round-trips.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>

#include "cholmod_batch.hpp"
#include "cholmod_solve.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using Int64Arr1D =
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Int64MutArr1D =
    nb::ndarray<nb::numpy, int64_t, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Float64Arr1D =
    nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Float64MutArr1D =
    nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Float64Arr2D =
    nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::device::cpu, nb::c_contig>;
using Float64MutArr2D =
    nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::device::cpu, nb::c_contig>;

NB_MODULE(_laplaciansolve_cholmod, m) {
  m.doc() = "Direct sparse Cholesky (CHOLMOD) backend for SPD systems.";
  m.attr("__has_cholmod__") = true;

  nb::class_<lsolve::CholmodSolver>(m, "CholmodSolver")
      .def(
          "__init__",
          [](lsolve::CholmodSolver* self, Int64Arr1D indptr, Int64Arr1D indices,
             Float64Arr1D data, int64_t n) {
            const std::size_t nnz = data.shape(0);
            nb::gil_scoped_release release;
            new (self) lsolve::CholmodSolver(indptr.data(), indices.data(),
                                             data.data(), static_cast<std::size_t>(n), nnz);
          },
          "indptr"_a, "indices"_a, "data"_a, "n"_a,
          "Analyze + factorize a symmetric SPD matrix (full sorted CSC, int64).")
      .def(
          "update",
          [](lsolve::CholmodSolver& self, Float64Arr1D data) {
            const std::size_t nnz = data.shape(0);
            const double* p = data.data();
            nb::gil_scoped_release release;
            self.update(p, nnz);
          },
          "data"_a, "Numeric refactorization with new values on the same pattern.")
      .def(
          "solve",
          [](lsolve::CholmodSolver& self, Float64Arr1D b, Float64MutArr1D x_out, int64_t k) {
            const double* pb = b.data();
            double* px = x_out.data();
            nb::gil_scoped_release release;
            self.solve(pb, px, static_cast<std::size_t>(k));
          },
          "b"_a, "x_out"_a, "k"_a,
          "Solve M x = b for k RHS (flat n*k buffers, [k, n] row-major).");

  // Batched factorization+solve over B SPD matrices sharing one pattern (PURC
  // per-destination): one shared analysis, B numeric factors + solves in one
  // GIL-released call, parallel over systems.  See BatchedCholmodSolver.
  nb::class_<lsolve::BatchedCholmodSolver>(m, "BatchedCholmodSolver")
      .def(
          "__init__",
          [](lsolve::BatchedCholmodSolver* self, Int64Arr1D indptr, Int64Arr1D indices,
             Float64Arr1D data, int64_t n) {
            const std::size_t nnz = data.shape(0);
            nb::gil_scoped_release release;
            new (self) lsolve::BatchedCholmodSolver(indptr.data(), indices.data(), data.data(),
                                                    static_cast<std::size_t>(n), nnz);
          },
          "indptr"_a, "indices"_a, "data"_a, "n"_a,
          "Analyze a symmetric SPD pattern once (full sorted CSC, int64); clones a "
          "symbolic factor per pool worker for the batched factorization.")
      .def("n", [](lsolve::BatchedCholmodSolver& self) { return static_cast<int64_t>(self.n()); },
           "System dimension.")
      .def("nnz",
           [](lsolve::BatchedCholmodSolver& self) { return static_cast<int64_t>(self.nnz()); },
           "Stored nonzeros of the shared CSC pattern.")
      .def(
          "factorize_and_solve_batch",
          [](lsolve::BatchedCholmodSolver& self, Float64Arr2D values, Float64Arr2D rhs,
             Int64Arr1D rhs_offsets, Float64MutArr2D x_out, Int64MutArr1D status_out) {
            const std::size_t batch = values.shape(0);
            const double* pv = values.data();
            const double* pr = rhs.data();
            const int64_t* po = rhs_offsets.data();
            double* px = x_out.data();
            int64_t* ps = status_out.data();
            nb::gil_scoped_release release;
            self.factorize_and_solve_batch(pv, batch, pr, po, px, ps);
          },
          "values"_a, "rhs"_a, "rhs_offsets"_a, "x_out"_a, "status_out"_a,
          "Per-system batch: values [B, nnz]; rhs/x_out flat [totalK, n] row-major with "
          "int64 rhs_offsets[B+1]; status_out [B] gets a per-system status (0 = ok).");
}
