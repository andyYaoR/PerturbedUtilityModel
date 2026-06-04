/**
 * nanobind bindings for the PURCSolver native core.
 *
 * M0 ships only the module skeleton plus a GIL-released smoke kernel that proves
 * the end-to-end scikit-build-core + nanobind build path on this machine.  All
 * arrays are borrowed zero-copy (numpy/scipy/torch-cpu) and the GIL is released
 * around compute, mirroring LaplacianSolve's nb_core_module.cpp.  The real PURC
 * kernels -- vectorized xi*(eta) recovery, fused weight/active-mask + CSC
 * assembly, and a full SSN step -- are added in M1+ behind the same `native`
 * import gate, with a pure-numpy fallback maintained for parity testing.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>

namespace nb = nanobind;
using namespace nb::literals;

// 1-D contiguous CPU array views (zero-copy).
using Float64Arr1D =
    nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Float64MutArr1D =
    nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Int64Arr1D =
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::device::cpu, nb::c_contig>;

namespace {

// In-place y += a * x.  Validates the zero-copy ndarray + GIL-release path that
// every real PURC kernel uses: pointers are taken under the GIL, then the GIL is
// dropped for the compute loop so concurrent solves scale across threads.
void axpy_f64(double a, Float64Arr1D x, Float64MutArr1D y) {
  const std::size_t n = x.shape(0);
  const double* xp = x.data();
  double* yp = y.data();
  nb::gil_scoped_release release;
  for (std::size_t i = 0; i < n; ++i) {
    yp[i] += a * xp[i];
  }
}

// Rectangular CSR sparse matrix-vector product y = A @ x (float64).
//
// A is (nrows x ncols) in CSR form (indptr length nrows+1, indices into x).
// y[i] = sum_{p in row i} data[p] * x[indices[p]].  y is overwritten.  This is
// the hot-path matvec used for A x_hat and A^T lambda every Newton iteration and
// every line-search dual evaluation; a dedicated GIL-released kernel keeps full
// float64 precision and avoids the overhead of the torch sparse-CSR (beta) path.
// Dependency-free (no Eigen / OpenMP, so no libomp coexistence issues with the
// LaplacianSolve / torch runtimes); -O3 -march=native vectorizes the inner loop.
void csr_spmv_f64(Int64Arr1D indptr, Int64Arr1D indices, Float64Arr1D data,
                  Float64Arr1D x, Float64MutArr1D y) {
  const int64_t nrows = static_cast<int64_t>(indptr.shape(0)) - 1;
  const int64_t* __restrict__ ip = indptr.data();
  const int64_t* __restrict__ ic = indices.data();
  const double* __restrict__ da = data.data();
  const double* __restrict__ xp = x.data();
  double* __restrict__ yp = y.data();
  nb::gil_scoped_release release;
  for (int64_t i = 0; i < nrows; ++i) {
    double s = 0.0;
    for (int64_t p = ip[i]; p < ip[i + 1]; ++p) {
      s += da[p] * xp[ic[p]];
    }
    yp[i] = s;
  }
}

}  // namespace

NB_MODULE(_purcsolver_core, m) {
  m.doc() = "PURCSolver native core (GIL-released hot-path kernels).";
  m.attr("__core_version__") = "0.1.0";
  m.def("axpy_f64", &axpy_f64, "a"_a, "x"_a, "y"_a,
        "In-place y += a*x (GIL released); smoke test of the native build path.");
  m.def("csr_spmv_f64", &csr_spmv_f64, "indptr"_a, "indices"_a, "data"_a, "x"_a,
        "y"_a,
        "Rectangular CSR SpMV y = A @ x (float64, GIL released); y overwritten.");
}
