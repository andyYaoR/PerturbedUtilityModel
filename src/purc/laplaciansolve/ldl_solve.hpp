// Triangular solve for the approxChol LDLinv factorization (the hot kernel).
//
// Faithful port of LDLsolver / forward! / backward! from Laplacians.jl
// (src/approxChol.jl), templated on index/value types.  This routine runs on
// every PCG iteration and every solve, so it is written for zero wasted work:
// flat array access in elimination order, fused multiplier+running-product
// updates, restrict-qualified pointers, and an optional skip of the null-space
// mean projection for nonsingular (SDDM) systems.
#pragma once

#include <cstddef>

#if defined(__GNUC__) || defined(__clang__)
#define LSOLVE_RESTRICT __restrict__
#else
#define LSOLVE_RESTRICT
#endif

namespace lsolve {

// Forward substitution: apply L^{-1} in place, walking the elimination order.
// Fuses the off-diagonal multiplier apply with the running yi *= (1 - f) update
// (matches approxChol.jl:1369-1388).
template <typename Tind, typename Tval>
inline void ldl_forward(const Tind* LSOLVE_RESTRICT col,
                        const Tind* LSOLVE_RESTRICT colptr,
                        const Tind* LSOLVE_RESTRICT rowval,
                        const Tval* LSOLVE_RESTRICT fval, std::size_t ncol,
                        Tval* LSOLVE_RESTRICT y) {
  for (std::size_t ii = 0; ii < ncol; ++ii) {
    const Tind i = col[ii];
    const Tind j0 = colptr[ii];
    const Tind j1 = colptr[ii + 1] - 1;  // index of this column's last entry
    Tval yi = y[i];
    for (Tind jj = j0; jj < j1; ++jj) {
      const Tind j = rowval[jj];
      const Tval f = fval[jj];
      y[j] += f * yi;
      yi *= (Tval(1) - f);
    }
    const Tind jlast = rowval[j1];
    y[jlast] += yi;
    y[i] = yi;
  }
}

// Backward substitution: apply L^{-T} in place, in reverse elimination order
// (matches approxChol.jl:1390-1408).  Tind must be signed.
template <typename Tind, typename Tval>
inline void ldl_backward(const Tind* LSOLVE_RESTRICT col,
                         const Tind* LSOLVE_RESTRICT colptr,
                         const Tind* LSOLVE_RESTRICT rowval,
                         const Tval* LSOLVE_RESTRICT fval, std::size_t ncol,
                         Tval* LSOLVE_RESTRICT y) {
  for (std::size_t k = ncol; k-- > 0;) {
    const Tind i = col[k];
    const Tind j0 = colptr[k];
    const Tind j1 = colptr[k + 1] - 1;
    const Tind jlast = rowval[j1];
    Tval yi = y[i] + y[jlast];
    for (Tind jj = j1 - 1; jj >= j0; --jj) {  // empty when the column has 1 entry
      const Tind j = rowval[jj];
      const Tval f = fval[jj];
      yi = (Tval(1) - f) * yi + f * y[j];
    }
    y[i] = yi;
  }
}

// Full solve: y = (L D L^T)^{-1} b.  Copies b into y, applies L^{-1}, the
// diagonal D^{-1} (skipping zero pivots), L^{-T}, then optionally projects out
// the constant null space (required for a singular Laplacian; skip for SDDM).
template <typename Tind, typename Tval>
inline void ldl_solve(const Tind* LSOLVE_RESTRICT col,
                      const Tind* LSOLVE_RESTRICT colptr,
                      const Tind* LSOLVE_RESTRICT rowval,
                      const Tval* LSOLVE_RESTRICT fval,
                      const Tval* LSOLVE_RESTRICT d, std::size_t ncol,
                      std::size_t n, const Tval* LSOLVE_RESTRICT b,
                      Tval* LSOLVE_RESTRICT y, bool subtract_mean) {
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = b[i];
  }
  ldl_forward<Tind, Tval>(col, colptr, rowval, fval, ncol, y);
  for (std::size_t i = 0; i < n; ++i) {
    if (d[i] != Tval(0)) {
      y[i] /= d[i];
    }
  }
  ldl_backward<Tind, Tval>(col, colptr, rowval, fval, ncol, y);
  if (subtract_mean) {
    Tval mu = Tval(0);
    for (std::size_t i = 0; i < n; ++i) {
      mu += y[i];
    }
    mu /= static_cast<Tval>(n);
    for (std::size_t i = 0; i < n; ++i) {
      y[i] -= mu;
    }
  }
}

}  // namespace lsolve
