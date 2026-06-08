// Host-callable entry points for the LaplacianSolve CUDA kernels.
//
// These are intentionally plain C++ signatures (no CUDA types) so the binding
// translation unit compiles without nvcc.  The bodies live in the sibling .cu
// files; until implemented on a GPU host they call raise_not_implemented().
// The full device memory-layout contract for each kernel is documented in
// docs/CUDA.md and in the corresponding .cu file.
#pragma once

namespace lsolve {
namespace cuda {

// Throw a clear "not implemented yet" error (defined in nb_cuda_module.cpp).
[[noreturn]] void raise_not_implemented(const char* name);

// Level-scheduled batched triangular solve of one shared LDLinv over B RHS.
void ldl_solve_batched_cuda();

// Batched CSC/CSR SpMV across B value-columns sharing one sparsity pattern.
void spmv_batched_cuda();

// Batched approxChol-preconditioned PCG run in lockstep over B systems.
void pcg_batched_cuda();

// One-warp-per-tree exact forest LDL solve (the acyclic optimum-phase path).
void forest_ldl_batched_cuda();

}  // namespace cuda
}  // namespace lsolve
