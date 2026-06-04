/**
 * nanobind bindings for the PURCSolver native core.
 *
 * M0 ships only the module skeleton plus a GIL-released smoke kernel that proves
 * the end-to-end scikit-build-core + nanobind build path on this machine.  All
 * arrays are borrowed zero-copy (numpy/scipy/torch-cpu) and the GIL is released
 * around compute, mirroring LaplacianSolve's nb_core_module.cpp.  The real PURC
 * kernels -- vectorized xi*(eta) recovery, fused weight/active-mask + CSC
 * assembly, and a full SSN step -- are added in M1+ behind the same `native`
 * import gate, with a pure-numpy fallback maintained for parity testing.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// 1-D contiguous CPU array views (zero-copy).
using Float64Arr1D =
    nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Float64MutArr1D =
    nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Int64Arr1D =
    nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Uint8MutArr1D =
    nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>, nb::device::cpu, nb::c_contig>;

namespace {

// In-place y += a * x.  Validates the zero-copy ndarray + GIL-release path that
// every real PURC kernel uses: pointers are taken under the GIL, then the GIL is
// dropped for the compute loop so concurrent solves scale across threads.
void axpy_f64(double a, Float64Arr1D x, Float64MutArr1D y) {
  const std::size_t n = x.shape(0);
  const double* xp = x.data();
  double* yp = y.data();
  nb::gil_scoped_release release;
  for (std::size_t i = 0; i < n; ++i) {
    yp[i] += a * xp[i];
  }
}

// Rectangular CSR sparse matrix-vector product y = A @ x (float64).
//
// A is (nrows x ncols) in CSR form (indptr length nrows+1, indices into x).
// y[i] = sum_{p in row i} data[p] * x[indices[p]].  y is overwritten.  This is
// the hot-path matvec used for A x_hat and A^T lambda every Newton iteration and
// every line-search dual evaluation; a dedicated GIL-released kernel keeps full
// float64 precision and avoids the overhead of the torch sparse-CSR (beta) path.
// Dependency-free (no Eigen / OpenMP, so no libomp coexistence issues with the
// LaplacianSolve / torch runtimes); -O3 -march=native vectorizes the inner loop.
void csr_spmv_f64(Int64Arr1D indptr, Int64Arr1D indices, Float64Arr1D data,
                  Float64Arr1D x, Float64MutArr1D y) {
  const int64_t nrows = static_cast<int64_t>(indptr.shape(0)) - 1;
  const int64_t* __restrict__ ip = indptr.data();
  const int64_t* __restrict__ ic = indices.data();
  const double* __restrict__ da = data.data();
  const double* __restrict__ xp = x.data();
  double* __restrict__ yp = y.data();
  nb::gil_scoped_release release;
  for (int64_t i = 0; i < nrows; ++i) {
    double s = 0.0;
    for (int64_t p = ip[i]; p < ip[i + 1]; ++p) {
      s += da[p] * xp[ic[p]];
    }
    yp[i] = s;
  }
}

// Fixed-pattern assembly of the Newton matrix values:
//   out[slot[p]] += coeff[p] * w[srci[p]]  (the A diag(w) A^T triple product),
//   then out[diag_slot[j]] += eps  (the eps*I regularizer).
// out has length nnz (the fixed CSC pattern) and is zeroed first.  This replaces
// the numpy bincount scatter in the per-iteration hot path.
void csc_assemble_f64(Int64Arr1D slot, Float64Arr1D coeff, Int64Arr1D srci,
                      Float64Arr1D w, Int64Arr1D diag_slot, double eps,
                      Float64MutArr1D out) {
  const int64_t ncontrib = static_cast<int64_t>(slot.shape(0));
  const int64_t kdiag = static_cast<int64_t>(diag_slot.shape(0));
  const int64_t nnz = static_cast<int64_t>(out.shape(0));
  const int64_t* __restrict__ sp = slot.data();
  const double* __restrict__ cp = coeff.data();
  const int64_t* __restrict__ si = srci.data();
  const double* __restrict__ wp = w.data();
  const int64_t* __restrict__ dp = diag_slot.data();
  double* __restrict__ op = out.data();
  nb::gil_scoped_release release;
  for (int64_t i = 0; i < nnz; ++i) op[i] = 0.0;
  for (int64_t p = 0; p < ncontrib; ++p) op[sp[p]] += cp[p] * wp[si[p]];
  for (int64_t j = 0; j < kdiag; ++j) op[dp[j]] += eps;
}

// Horner evaluation of a polynomial with coefficients c[0..deg] (low to high).
inline double horner(const double* c, int64_t deg, double x) {
  double s = c[deg];
  for (int64_t k = deg - 1; k >= 0; --k) s = s * x + c[k];
  return s;
}

// Vectorized inverse of a strictly-increasing polynomial h'(xi) = eta on the box
// [lo, hi], by safeguarded Newton ("rtsafe").  coeffs are the h' coefficients
// (low to high); the derivative h'' is taken analytically.  Saturated coords
// clip to the bound (interior=0); interior coords solve the unique bracketed
// root.  This is the parameterized native recovery kernel for the sieve.
void recovery_poly_f64(Float64Arr1D coeffs, Float64Arr1D eta, Float64Arr1D lo,
                       Float64Arr1D hi, Float64MutArr1D xi_out,
                       Uint8MutArr1D interior_out, int64_t max_iter,
                       double xtol) {
  const int64_t deg = static_cast<int64_t>(coeffs.shape(0)) - 1;
  const int64_t n = static_cast<int64_t>(eta.shape(0));
  const double* cc = coeffs.data();
  const double* ep = eta.data();
  const double* lp = lo.data();
  const double* hp = hi.data();
  double* xp = xi_out.data();
  uint8_t* ip = interior_out.data();

  // Derivative coefficients dc[k] = (k+1) c[k+1], degree deg-1.
  std::vector<double> dc(deg > 0 ? static_cast<std::size_t>(deg) : 1, 0.0);
  for (int64_t k = 0; k < deg; ++k) dc[k] = (k + 1) * cc[k + 1];
  const double* dcp = dc.data();
  const int64_t ddeg = deg - 1;

  nb::gil_scoped_release release;
  for (int64_t i = 0; i < n; ++i) {
    const double L = lp[i], H = hp[i], t = ep[i];
    const double gL = horner(cc, deg, L), gH = horner(cc, deg, H);
    if (t <= gL) {
      xp[i] = L;
      ip[i] = 0;
      continue;
    }
    if (t >= gH) {
      xp[i] = H;
      ip[i] = 0;
      continue;
    }
    double a = L, b = H;
    double x = L + (t - gL) / (gH - gL) * (H - L);
    for (int64_t it = 0; it < max_iter; ++it) {
      const double f = horner(cc, deg, x) - t;
      if (f > 0.0) b = x; else a = x;
      const double df = (ddeg >= 0) ? horner(dcp, ddeg, x) : 0.0;
      double xn = x - f / df;
      if (!(xn > a && xn < b) || !std::isfinite(xn)) xn = 0.5 * (a + b);
      const double step = xn - x;
      x = xn;
      if (std::fabs(step) <= xtol * (1.0 + std::fabs(x))) break;
    }
    xp[i] = x;
    ip[i] = (x > L && x < H) ? 1 : 0;
  }
}

// h'(x) and h''(x) for the barrier recovery, dispatched by `kernel`:
//   0 quadratic, 1 Shannon entropy, 2 logit/binary entropy, 3 modified entropy,
//   4 polynomial sieve (h' coefficients in `cc`, low->high; h'' in `dc`).
inline double bk_hp(int64_t kernel, const double* cc, int64_t deg, double x) {
  switch (kernel) {
    case 0: return x;
    case 1: return 1.0 + std::log(x);
    case 2: return std::log(x) - std::log1p(-x);
    case 3: return std::log1p(x);
    default: return horner(cc, deg, x);
  }
}
inline double bk_hpp(int64_t kernel, const double* dc, int64_t ddeg, double x) {
  switch (kernel) {
    case 0: return 1.0;
    case 1: return 1.0 / x;
    case 2: return 1.0 / (x * (1.0 - x));
    case 3: return 1.0 / (1.0 + x);
    default: return (ddeg >= 0) ? horner(dc, ddeg, x) : 0.0;
  }
}

// Per-coordinate barrier-smoothed primal recovery.  For each i, solve the
// strictly-monotone scalar root
//   q(x) = ell*h'(x) - y - mu/(x-lo) + mu/(hi-x) = 0   on the open box (lo, hi)
// by safeguarded Newton, and return x and the Schur weight 1/q'(x), where
//   q'(x) = ell*h''(x) + mu/(x-lo)^2 + mu/(hi-x)^2 > 0.
// Unlike the vectorized torch root-find, every coordinate converges on its OWN
// Newton schedule (no batch-wide `all(width<=tol)` gate, no shared bisection
// cap), which is the whole point: the stiff entropy fallback stops running every
// coordinate to max_iter.  `x0` is the per-coordinate warm start (clamped into
// the safeguarded bracket [lo+delta, hi-delta]).
void recover_barrier_f64(int64_t kernel, Float64Arr1D coeffs, Float64Arr1D ell,
                         Float64Arr1D y, Float64Arr1D lo, Float64Arr1D hi,
                         Float64Arr1D x0, double mu, double endpoint_margin,
                         Float64MutArr1D x_out, Float64MutArr1D w_out,
                         int64_t max_iter, double xtol) {
  const int64_t n = static_cast<int64_t>(y.shape(0));
  const int64_t deg = static_cast<int64_t>(coeffs.shape(0)) - 1;
  const double* cc = coeffs.data();
  const double* ellp = ell.data();
  const double* yp = y.data();
  const double* lp = lo.data();
  const double* hp = hi.data();
  const double* x0p = x0.data();
  double* xo = x_out.data();
  double* wo = w_out.data();

  // sieve h'' coefficients dc[k] = (k+1) c[k+1], degree deg-1.
  std::vector<double> dc(deg > 0 ? static_cast<std::size_t>(deg) : 1, 0.0);
  for (int64_t k = 0; k < deg; ++k) dc[k] = (k + 1) * cc[k + 1];
  const double* dcp = dc.data();
  const int64_t ddeg = deg - 1;
  const double tiny = 2.220446049250313e-16;

  nb::gil_scoped_release release;
  for (int64_t i = 0; i < n; ++i) {
    const double L = lp[i], H = hp[i], el = ellp[i], yi = yp[i];
    const double gap = H - L;
    double delta = gap * endpoint_margin;
    if (delta < tiny) delta = tiny;
    if (delta > 0.25 * gap) delta = 0.25 * gap;
    double a = L + delta, b = H - delta;
    double x = x0p[i];
    if (!(x > a)) x = a;
    if (!(x < b)) x = b;
    // Safeguarded Newton with a BRACKET-WIDTH stopping test.  A step-size test is
    // unsafe here: near a bound q'(x)=ell*h''+mu/(x-lo)^2+mu/(hi-x)^2 blows up, so
    // the Newton step -q/q' is tiny even far from the root -- a step test would
    // stop at the wrong point.  Bracketing on the sign of the monotone q is the
    // robust criterion (matches the torch reference, per-coordinate).
    for (int64_t it = 0; it < max_iter; ++it) {
      const double xl = x - L, xh = H - x;
      const double q = el * bk_hp(kernel, cc, deg, x) - yi - mu / xl + mu / xh;
      if (q > 0.0) b = x; else a = x;
      const double qp =
          el * bk_hpp(kernel, dcp, ddeg, x) + mu / (xl * xl) + mu / (xh * xh);
      double xn = x - q / qp;
      if (!(xn > a && xn < b) || !std::isfinite(xn)) xn = 0.5 * (a + b);
      x = xn;
      if ((b - a) <= xtol * (1.0 + gap)) break;
    }
    const double xl = x - L, xh = H - x;
    const double qp =
        el * bk_hpp(kernel, dcp, ddeg, x) + mu / (xl * xl) + mu / (xh * xh);
    xo[i] = x;
    wo[i] = 1.0 / qp;
  }
}

}  // namespace

NB_MODULE(_purcsolver_core, m) {
  m.doc() = "PURCSolver native core (GIL-released hot-path kernels).";
  m.attr("__core_version__") = "0.1.0";
  m.def("axpy_f64", &axpy_f64, "a"_a, "x"_a, "y"_a,
        "In-place y += a*x (GIL released); smoke test of the native build path.");
  m.def("csr_spmv_f64", &csr_spmv_f64, "indptr"_a, "indices"_a, "data"_a, "x"_a,
        "y"_a,
        "Rectangular CSR SpMV y = A @ x (float64, GIL released); y overwritten.");
  m.def("csc_assemble_f64", &csc_assemble_f64, "slot"_a, "coeff"_a, "srci"_a,
        "w"_a, "diag_slot"_a, "eps"_a, "out"_a,
        "Assemble Newton matrix values A diag(w) A^T + eps I (GIL released).");
  m.def("recovery_poly_f64", &recovery_poly_f64, "coeffs"_a, "eta"_a, "lo"_a,
        "hi"_a, "xi_out"_a, "interior_out"_a, "max_iter"_a = 80,
        "xtol"_a = 1e-14,
        "Invert a monotone polynomial h'(xi)=eta on [lo,hi] (GIL released).");
  m.def("recover_barrier_f64", &recover_barrier_f64, "kernel"_a, "coeffs"_a,
        "ell"_a, "y"_a, "lo"_a, "hi"_a, "x0"_a, "mu"_a, "endpoint_margin"_a,
        "x_out"_a, "w_out"_a, "max_iter"_a = 80, "xtol"_a = 1e-13,
        "Per-coordinate barrier-smoothed primal recovery x and weight 1/q' "
        "(GIL released); kernel 0=quad 1=entropy 2=logit 3=mod-entropy 4=sieve.");
}
