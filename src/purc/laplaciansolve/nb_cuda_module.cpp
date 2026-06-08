/**
 * nanobind bindings for the LaplacianSolve CUDA module (_laplaciansolve_cuda).
 *
 * This translation unit is compiled only on hosts with a CUDA compiler (see
 * src/laplaciansolve/CMakeLists.txt).  It registers the same logical entry
 * points as the CPU core, but the kernel bodies in cuda/*.cu are placeholders
 * that raise until implemented on a GPU machine (Stage 4; see docs/CUDA.md).
 *
 * The module importing successfully signals "CUDA build present"; calling any
 * kernel before it is implemented raises a clear RuntimeError - never a silent
 * fallback.
 */

#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>

#include "cuda/kernels.hpp"

namespace nb = nanobind;

namespace lsolve {
namespace cuda {

void raise_not_implemented(const char* name) {
  throw std::runtime_error(std::string("LaplacianSolve CUDA kernel '") + name +
                           "' is not implemented yet; fill in "
                           "src/laplaciansolve/cuda/ on a GPU host (see docs/CUDA.md).");
}

}  // namespace cuda
}  // namespace lsolve

NB_MODULE(_laplaciansolve_cuda, m) {
  m.doc() = "LaplacianSolve CUDA module (kernel bodies pending; see docs/CUDA.md).";
  m.attr("__has_cuda_build__") = true;

  m.def("ldl_solve_batched", []() { lsolve::cuda::ldl_solve_batched_cuda(); },
        "Level-scheduled batched LDLinv solve (not yet implemented).");
  m.def("spmv_batched", []() { lsolve::cuda::spmv_batched_cuda(); },
        "Batched shared-pattern SpMV (not yet implemented).");
  m.def("pcg_batched", []() { lsolve::cuda::pcg_batched_cuda(); },
        "Batched approxChol-preconditioned PCG (not yet implemented).");
  m.def("forest_ldl_batched", []() { lsolve::cuda::forest_ldl_batched_cuda(); },
        "One-warp-per-tree exact forest solve (not yet implemented).");
}
