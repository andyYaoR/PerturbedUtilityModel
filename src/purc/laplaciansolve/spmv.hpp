// Sparse matrix-vector product for CSC/CSR matrices.
//
// y = A @ x.  For the symmetric Laplacians this solver targets, CSC and CSR are
// identical, so a single column-walk kernel serves the PCG matvec.
#pragma once

#include <cstddef>

#include "csc_matrix.hpp"

namespace lsolve {

// y = A @ x, accumulating column j's contribution x[j] * A[:, j].
template <typename Tind, typename Tval>
inline void csc_matvec(const Tind* indptr, const Tind* indices, const Tval* data,
                       std::size_t ncols, std::size_t nrows, const Tval* x,
                       Tval* y) {
  for (std::size_t i = 0; i < nrows; ++i) {
    y[i] = Tval(0);
  }
  for (std::size_t j = 0; j < ncols; ++j) {
    const Tval xj = x[j];
    const Tind end = indptr[j + 1];
    for (Tind p = indptr[j]; p < end; ++p) {
      y[indices[p]] += data[p] * xj;
    }
  }
}

// y = A @ x via a row walk (CSR).  For a SYMMETRIC matrix (a graph Laplacian)
// the CSC arrays are also valid CSR, so this gives the same result as
// csc_matvec.  Serial: it runs inside one PCG system, and batch parallelism is
// applied one level up (one thread per independent system), so threading the row
// walk too would only oversubscribe.
template <typename Tind, typename Tval>
inline void csr_matvec(const Tind* indptr, const Tind* indices, const Tval* data,
                       std::size_t nrows, const Tval* x, Tval* y) {
  for (std::size_t i = 0; i < nrows; ++i) {
    Tval s = Tval(0);
    const Tind end = indptr[i + 1];
    for (Tind p = indptr[i]; p < end; ++p) {
      s += data[p] * x[indices[p]];
    }
    y[i] = s;
  }
}

template <typename Tind, typename Tval>
inline void csc_matvec(const CscView<Tind, Tval>& a, const Tval* x, Tval* y) {
  csc_matvec<Tind, Tval>(a.indptr, a.indices, a.data, a.ncols, a.nrows, x, y);
}

}  // namespace lsolve
