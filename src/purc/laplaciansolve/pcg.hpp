// Preconditioned conjugate gradient (and CG), templated on matvec/precond.
//
// Faithful port of pcg from Laplacians.jl (src/pcg.jl): best-residual iterate
// tracking, the stag_test stagnation guard, and the periodic stagnation
// heuristics.  The two divisions that can hit the Laplacian null space are
// guarded (Julia tolerates them via float Inf semantics; we stop cleanly),
// matching the Stage-0 Python reference.
//
// MatVec and Precond are callables with signature void(const Tval* in, Tval* out)
// over length-n vectors, so the same driver serves any operator/preconditioner.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace lsolve {

template <typename Tval>
struct PcgResult {
  int iterations;
  Tval relres;  // best relative residual ||b - A x|| / ||b||
  bool converged;
};

namespace detail {

// These vector kernels are serial: they run inside one PCG system, and batch
// parallelism is applied one level up (one thread per independent system), so
// threading them too would only oversubscribe.

template <typename Tval>
inline Tval dot(const Tval* a, const Tval* b, std::size_t n) {
  Tval s = Tval(0);
  for (std::size_t i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

template <typename Tval>
inline Tval norm(const Tval* a, std::size_t n) {
  return std::sqrt(dot(a, a, n));
}

template <typename Tval>
inline Tval abs_sum(const Tval* a, std::size_t n) {
  Tval s = Tval(0);
  for (std::size_t i = 0; i < n; ++i) {
    s += std::abs(a[i]);
  }
  return s;
}

// y += a * x  (axpy).
template <typename Tval>
inline void axpy(Tval alpha, const Tval* x, Tval* y, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

}  // namespace detail

// Solve A x = b with preconditioner M^{-1}.  x_out must be presized to n and is
// also the warm-start (pass a zero vector for a cold start).
template <typename Tval, typename MatVec, typename Precond>
PcgResult<Tval> pcg(std::size_t n, const Tval* b, MatVec matvec, Precond precond,
                    Tval tol, int maxits, int stag_test, Tval* x_out) {
  const Tval eps = std::numeric_limits<Tval>::epsilon();
  const Tval nb = detail::norm(b, n);
  if (nb == Tval(0)) {
    for (std::size_t i = 0; i < n; ++i) {
      x_out[i] = Tval(0);
    }
    return {0, Tval(0), true};
  }

  std::vector<Tval> r(n), z(n), p(n), q(n), bestx(n);

  // Residual from the (possibly warm-started) x_out.
  bool all_zero = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (x_out[i] != Tval(0)) {
      all_zero = false;
      break;
    }
  }
  if (all_zero) {
    for (std::size_t i = 0; i < n; ++i) {
      r[i] = b[i];
    }
  } else {
    matvec(x_out, q.data());  // q = A x0
    for (std::size_t i = 0; i < n; ++i) {
      r[i] = b[i] - q[i];
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    bestx[i] = x_out[i];
  }
  Tval bestnr = detail::norm(r.data(), n) / nb;

  precond(r.data(), z.data());
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = z[i];
  }
  Tval rho = detail::dot(r.data(), z.data(), n);
  Tval best_rho = rho;
  int stag_count = 0;

  int itcnt = 0;
  bool converged = bestnr < tol;
  while (itcnt < maxits && !converged) {
    ++itcnt;

    matvec(p.data(), q.data());
    const Tval pq = detail::dot(p.data(), q.data(), n);
    // PSD curvature: pq <= 0 means p reached the null space (converged).
    if (!std::isfinite(pq) || pq <= Tval(0)) {
      break;
    }
    const Tval al = rho / pq;

    if (itcnt % 10 == 0 &&
        al * detail::abs_sum(p.data(), n) < eps * detail::abs_sum(x_out, n)) {
      break;
    }
    detail::axpy(al, p.data(), x_out, n);  // x += al * p
    if (itcnt % 10 == 0 &&
        al * detail::abs_sum(q.data(), n) < eps * detail::abs_sum(r.data(), n)) {
      break;
    }
    detail::axpy(-al, q.data(), r.data(), n);  // r -= al * q

    const Tval nr = detail::norm(r.data(), n) / nb;
    if (nr < bestnr) {
      bestnr = nr;
      for (std::size_t i = 0; i < n; ++i) {
        bestx[i] = x_out[i];
      }
    }
    if (nr < tol) {
      converged = true;
      break;
    }

    precond(r.data(), z.data());
    const Tval oldrho = rho;
    rho = detail::dot(z.data(), r.data(), n);

    if (stag_test > 0) {
      const Tval threshold = best_rho * (Tval(1) - Tval(1) / stag_test);
      if (rho < threshold) {
        best_rho = rho;
        stag_count = 0;
      } else if (best_rho > (Tval(1) - Tval(1) / stag_test) * rho) {
        ++stag_count;
        if (stag_count > stag_test) {
          break;
        }
      }
    }

    if (std::isinf(rho)) {
      break;
    }
    if (oldrho == Tval(0)) {  // previous rho collapsed to zero => converged
      break;
    }
    const Tval beta = rho / oldrho;
    if (std::isinf(beta)) {
      break;
    }
    if (itcnt % 10 == 0 &&
        beta * detail::abs_sum(p.data(), n) < eps * detail::abs_sum(z.data(), n)) {
      break;
    }
    for (std::size_t i = 0; i < n; ++i) {  // p = z + beta * p
      p[i] = z[i] + beta * p[i];
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    x_out[i] = bestx[i];
  }
  return {itcnt, bestnr, bestnr < tol};
}

}  // namespace lsolve
