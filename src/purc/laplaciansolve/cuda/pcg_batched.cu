// CUDA: batched approxChol-preconditioned PCG.  [STUB - see docs/CUDA.md]
//
// Goal: run B independent PCG solves in lockstep on the GPU - the main win for
// PURC, where each outer step solves B OD-pair systems that share the fixed
// sparsity pattern and converge in a handful of iterations.
//
// Composition:
//   matvec  : spmv_batched_cuda (B systems, shared pattern)
//   precond : ldl_solve_batched_cuda (one shared LDLinv, or B per-system ones)
//   vector ops (axpy, dot, norm): batched/segmented reductions over [B, n],
//     one reduction per system so all B advance together.
//
// Device memory-layout contract: the SpMV and LDL-solve contracts above, plus
//   b/x_out : float64[B, n]   (x_out warm-starts and receives the solution)
//   tol, maxits, stag_test : per-call scalars (shared across the batch)
//   returns per-system (iterations, relres, converged).
//
// Keep per-system convergence flags so a converged system stops contributing;
// iterate until all converge or maxits.  Parity target: per-system results
// match the CPU pcg within tolerance(dtype) and batched == looped (tests).

#include "kernels.hpp"

namespace lsolve {
namespace cuda {

void pcg_batched_cuda() { raise_not_implemented("pcg_batched"); }

}  // namespace cuda
}  // namespace lsolve
