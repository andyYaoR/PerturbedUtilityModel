// CUDA: batched CSC/CSR SpMV across B value-columns.  [STUB - see docs/CUDA.md]
//
// Goal: y[b] = A[b] @ x[b] for B systems that SHARE one sparsity pattern
// (indptr/indices) but differ in their values - exactly the PURC case where the
// full-graph pattern is fixed and only weights change per OD pair / iteration.
//
// Device memory-layout contract (row-major, device pointers):
//   indptr   : int64[n + 1]           shared CSC/CSR column pointers
//   indices  : int64[nnz]             shared row indices
//   data     : float64[B, nnz]        per-system values (B contiguous blocks)
//   x        : float64[B, n]          input vectors
//   y_out    : float64[B, n]          output vectors
//
// Kernel plan: one warp per (system, row) accumulating the row dot-product, or
// a CSR-vector kernel with coalesced loads over the shared `indices`.  Because
// the pattern is shared, `indices`/`indptr` stay resident and only `data`/`x`
// stream per system - good arithmetic intensity for the batched PCG matvec.
// Parity target: matches CPU csc_matvec per system within tolerance(dtype).

#include "kernels.hpp"

namespace lsolve {
namespace cuda {

void spmv_batched_cuda() { raise_not_implemented("spmv_batched"); }

}  // namespace cuda
}  // namespace lsolve
