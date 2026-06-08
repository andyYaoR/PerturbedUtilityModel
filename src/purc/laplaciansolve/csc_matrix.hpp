// Lightweight non-owning CSC matrix view over caller-owned buffers.
//
// The core never owns matrix storage; it borrows the (indptr, indices, data)
// arrays supplied by NumPy/SciPy/torch through the bindings.  For a symmetric
// matrix (a graph Laplacian) CSC and CSR coincide, so the same view serves both.
#pragma once

#include <cstddef>

namespace lsolve {

template <typename Tind, typename Tval>
struct CscView {
  const Tind* indptr;   // length ncols + 1
  const Tind* indices;  // row indices, length nnz
  const Tval* data;     // values, length nnz
  std::size_t nrows;
  std::size_t ncols;

  CscView(const Tind* indptr_, const Tind* indices_, const Tval* data_,
          std::size_t nrows_, std::size_t ncols_)
      : indptr(indptr_),
        indices(indices_),
        data(data_),
        nrows(nrows_),
        ncols(ncols_) {}
};

}  // namespace lsolve
