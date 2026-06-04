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

namespace {

// In-place y += a * x.  Validates the zero-copy ndarray + GIL-release path that
// every real PURC kernel will use: pointers are taken under the GIL, then the
// GIL is dropped for the compute loop so concurrent solves scale across threads.
void axpy_f64(double a, Float64Arr1D x, Float64MutArr1D y) {
  const std::size_t n = x.shape(0);
  const double* xp = x.data();
  double* yp = y.data();
  nb::gil_scoped_release release;
  for (std::size_t i = 0; i < n; ++i) {
    yp[i] += a * xp[i];
  }
}

}  // namespace

NB_MODULE(_purcsolver_core, m) {
  m.doc() = "PURCSolver native core (M0 skeleton: GIL-released smoke kernel).";
  m.attr("__core_version__") = "0.0.0";
  m.def("axpy_f64", &axpy_f64, "a"_a, "x"_a, "y"_a,
        "In-place y += a*x (GIL released); smoke test of the native build path.");
}
