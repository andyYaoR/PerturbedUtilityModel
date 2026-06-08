// CUDA: level-scheduled batched LDLinv triangular solve.  [STUB - see docs/CUDA.md]
//
// Goal: apply one shared approxChol LDLinv to a batch of B right-hand sides on
// the GPU.  The approxChol BUILD stays on the CPU (it is sequential); the GPU
// accelerates the solve, which dominates PCG cost.
//
// Device memory-layout contract (all device pointers, row-major):
//   col      : int64[ncol]            elimination order
//   colptr   : int64[ncol + 1]        per-column offsets into rowval/fval
//   rowval   : int64[nnzL]            L row indices
//   fval     : float64[nnzL]          L multipliers
//   d        : float64[n]             diagonal (0 marks a null pivot)
//   levels   : int32[ncol]            level index per eliminated column,
//              computed CPU-side at upload time; columns in the same level have
//              no data dependency and may run concurrently
//   level_ptr: int32[nlevels + 1]     CSR-style grouping of columns by level
//   b/x_out  : float64[B, n]          batched RHS / solution (B = batch size)
//
// Kernel plan:
//   forward sweep  : for each level in increasing order, one thread-block per
//                    (system, column) applies the fused multiplier + running
//                    product update; sync between levels.
//   diagonal scale : elementwise y /= d (skip d == 0).
//   backward sweep : levels in decreasing order, mirror of forward.
//   optional mean-subtract per system for the singular Laplacian case.
// Parity target: matches the CPU ldl_solve within tolerance(dtype)
//                (tests/test_cuda.py).

#include "kernels.hpp"

namespace lsolve {
namespace cuda {

void ldl_solve_batched_cuda() { raise_not_implemented("ldl_solve_batched"); }

}  // namespace cuda
}  // namespace lsolve
