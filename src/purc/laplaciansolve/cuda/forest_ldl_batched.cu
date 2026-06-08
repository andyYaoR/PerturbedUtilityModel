// CUDA: one-warp-per-tree exact forest LDL solve.  [STUB - see docs/CUDA.md]
//
// Goal: at the PURC optimum the active subgraph is a FOREST, so the Laplacian
// factorizes exactly with no fill-in and the solve is O(|S|).  This kernel
// solves many small trees at once - the common, fast optimum-phase case.
//
// Device memory-layout contract (device pointers):
//   parent   : int64[n]        parent in the rooted forest (-1 for roots),
//                              in a leaf->root elimination order
//   pweight  : float64[n]      edge weight to parent (0 for roots)
//   order    : int64[n]        leaf-pruning elimination order
//   tree_id  : int32[n]        component/tree index per vertex
//   b/x_out  : float64[B, n]   batched RHS / solution
//
// Kernel plan: assign one warp (or block) per (system, tree); sweep the
// elimination order to push each leaf's residual to its parent (forward), then
// back-substitute root->leaf, subtracting the per-tree mean for consistency.
// No PCG and no randomness - this is exact.  Parity target: matches the CPU
// forest_solve / exact dense solve within tolerance(dtype).

#include "kernels.hpp"

namespace lsolve {
namespace cuda {

void forest_ldl_batched_cuda() { raise_not_implemented("forest_ldl_batched"); }

}  // namespace cuda
}  // namespace lsolve
