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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
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
using Uint8Arr1D =
    nb::ndarray<nb::numpy, const uint8_t, nb::ndim<1>, nb::device::cpu, nb::c_contig>;

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

// Per-system complementarity mu and centrality ratio cr over a trial (xt,zt,wt)
// row of length n: mu = (sum xt*zt + (1-xt)*wt)/(2n); cr = min_i product_i / mu.
static inline void mu_cr_row(const double* xt, const double* zt, const double* wt,
                             int64_t n, double& mu_out, double& cr_out) {
  double sump = 0.0;
  double pmin = std::numeric_limits<double>::infinity();
  for (int64_t i = 0; i < n; ++i) {
    const double pz = xt[i] * zt[i];
    const double pw = (1.0 - xt[i]) * wt[i];
    sump += pz + pw;
    if (pz < pmin) pmin = pz;
    if (pw < pmin) pmin = pw;
  }
  const double mu = sump / (2.0 * static_cast<double>(n));
  mu_out = mu;
  cr_out = (mu > 0.0) ? pmin / mu : 0.0;
}

// Fused fast-step backtracking (the batched IPM safeguard, the rho-decrease arm).
// Mirrors the torch tau-schedule loop exactly: from the FROZEN base (x,z,w,lam)
// and corrector directions, find per OD the largest tau in {1, chi, chi^2, ...}
// (<= ls_max trials) at which the trial point first satisfies mu(tau) <= rho*mu and
// centrality cr >= gamma_min, accept+freeze it, and leave already-done systems
// (the converged ones, via done_in) untouched.  Outputs xf,zf,wf,lamf are written
// from the base and updated only on acceptance; done_out flags the accepted set.
void ipm_fast_backtrack_f64(Float64Arr1D x, Float64Arr1D z, Float64Arr1D w,
                            Float64Arr1D lam, Float64Arr1D dx, Float64Arr1D dz,
                            Float64Arr1D dw, Float64Arr1D dlam,
                            Float64Arr1D ap_max, Float64Arr1D ad_max,
                            Float64Arr1D mu, Uint8Arr1D done_in, double rho,
                            double gamma_min, double chi, int64_t ls_max, int64_t B,
                            int64_t N, int64_t k, Float64MutArr1D xf,
                            Float64MutArr1D zf, Float64MutArr1D wf,
                            Float64MutArr1D lamf, Uint8MutArr1D done_out) {
  const double* xp = x.data();   const double* zp = z.data();
  const double* wp = w.data();   const double* lamp = lam.data();
  const double* dxp = dx.data(); const double* dzp = dz.data();
  const double* dwp = dw.data(); const double* dlamp = dlam.data();
  const double* apm = ap_max.data(); const double* adm = ad_max.data();
  const double* mup = mu.data();     const uint8_t* din = done_in.data();
  double* xfo = xf.data(); double* zfo = zf.data(); double* wfo = wf.data();
  double* lfo = lamf.data(); uint8_t* dout = done_out.data();

  nb::gil_scoped_release release;
  for (int64_t b = 0; b < B; ++b) {
    dout[b] = din[b];
    for (int64_t i = 0; i < N; ++i) {
      xfo[b * N + i] = xp[b * N + i];
      zfo[b * N + i] = zp[b * N + i];
      wfo[b * N + i] = wp[b * N + i];
    }
    for (int64_t i = 0; i < k; ++i) lfo[b * k + i] = lamp[b * k + i];
  }
  std::vector<double> xt(static_cast<std::size_t>(N));
  std::vector<double> zt(static_cast<std::size_t>(N));
  std::vector<double> wt(static_cast<std::size_t>(N));
  double tau = 1.0;
  for (int64_t it = 0; it < ls_max; ++it) {
    bool all_done = true;
    for (int64_t b = 0; b < B; ++b) {
      if (dout[b]) continue;
      const double ap = tau * apm[b], ad = tau * adm[b];
      const double* xb = xp + b * N; const double* zb = zp + b * N;
      const double* wb = wp + b * N; const double* dxb = dxp + b * N;
      const double* dzb = dzp + b * N; const double* dwb = dwp + b * N;
      for (int64_t i = 0; i < N; ++i) {
        xt[i] = xb[i] + ap * dxb[i];
        zt[i] = zb[i] + ad * dzb[i];
        wt[i] = wb[i] + ad * dwb[i];
      }
      double mut, cr;
      mu_cr_row(xt.data(), zt.data(), wt.data(), N, mut, cr);
      const bool ok = (mut <= rho * mup[b]) && (cr >= gamma_min);
      if (ok) {
        for (int64_t i = 0; i < N; ++i) {
          xfo[b * N + i] = xt[i]; zfo[b * N + i] = zt[i]; wfo[b * N + i] = wt[i];
        }
        for (int64_t i = 0; i < k; ++i)
          lfo[b * k + i] = lamp[b * k + i] + ad * dlamp[b * k + i];
        dout[b] = 1;
      } else {
        all_done = false;
      }
    }
    if (all_done) break;
    tau *= chi;
  }
}

// Fused safe-step backtracking (the Armijo arm of the safeguard).  Mirrors the
// torch loop: from the base and the centred safe directions, update every still
// pending OD to the LATEST trial each tau (not only on acceptance), and accept when
// mu(tau) <= (1 - kappa*tau*(1-sigma_s))*mu and cr >= gamma_min.  Systems flagged
// done_in (those that do NOT need a safe step) are left at the base.
void ipm_safe_backtrack_f64(Float64Arr1D x, Float64Arr1D z, Float64Arr1D w,
                            Float64Arr1D lam, Float64Arr1D dxs, Float64Arr1D dzs,
                            Float64Arr1D dws, Float64Arr1D dlams,
                            Float64Arr1D ap_s, Float64Arr1D ad_s, Float64Arr1D mu,
                            Float64Arr1D sigma_s, Uint8Arr1D done_in, double kappa,
                            double gamma_min, double chi, int64_t ls_max, int64_t B,
                            int64_t N, int64_t k, Float64MutArr1D xs,
                            Float64MutArr1D zs, Float64MutArr1D ws,
                            Float64MutArr1D lams, Uint8MutArr1D done_out) {
  const double* xp = x.data();   const double* zp = z.data();
  const double* wp = w.data();   const double* lamp = lam.data();
  const double* dxp = dxs.data(); const double* dzp = dzs.data();
  const double* dwp = dws.data(); const double* dlamp = dlams.data();
  const double* apm = ap_s.data(); const double* adm = ad_s.data();
  const double* mup = mu.data();   const double* sig = sigma_s.data();
  const uint8_t* din = done_in.data();
  double* xso = xs.data(); double* zso = zs.data(); double* wso = ws.data();
  double* lso = lams.data(); uint8_t* dout = done_out.data();

  nb::gil_scoped_release release;
  for (int64_t b = 0; b < B; ++b) {
    dout[b] = din[b];
    for (int64_t i = 0; i < N; ++i) {
      xso[b * N + i] = xp[b * N + i];
      zso[b * N + i] = zp[b * N + i];
      wso[b * N + i] = wp[b * N + i];
    }
    for (int64_t i = 0; i < k; ++i) lso[b * k + i] = lamp[b * k + i];
  }
  std::vector<double> xt(static_cast<std::size_t>(N));
  std::vector<double> zt(static_cast<std::size_t>(N));
  std::vector<double> wt(static_cast<std::size_t>(N));
  double tau = 1.0;
  for (int64_t it = 0; it < ls_max; ++it) {
    bool all_done = true;
    for (int64_t b = 0; b < B; ++b) {
      if (dout[b]) continue;
      const double ap = tau * apm[b], ad = tau * adm[b];
      const double* xb = xp + b * N; const double* zb = zp + b * N;
      const double* wb = wp + b * N; const double* dxb = dxp + b * N;
      const double* dzb = dzp + b * N; const double* dwb = dwp + b * N;
      for (int64_t i = 0; i < N; ++i) {
        xt[i] = xb[i] + ap * dxb[i];
        zt[i] = zb[i] + ad * dzb[i];
        wt[i] = wb[i] + ad * dwb[i];
      }
      // Update on pending (the latest trial), then test Armijo acceptance.
      for (int64_t i = 0; i < N; ++i) {
        xso[b * N + i] = xt[i]; zso[b * N + i] = zt[i]; wso[b * N + i] = wt[i];
      }
      for (int64_t i = 0; i < k; ++i)
        lso[b * k + i] = lamp[b * k + i] + ad * dlamp[b * k + i];
      double mut, cr;
      mu_cr_row(xt.data(), zt.data(), wt.data(), N, mut, cr);
      const bool armijo = mut <= (1.0 - kappa * tau * (1.0 - sig[b])) * mup[b];
      const bool ok = armijo && (cr >= gamma_min);
      if (ok) dout[b] = 1; else all_done = false;
    }
    if (all_done) break;
    tau *= chi;
  }
}

// Dense solve M x = rhs (n x n row-major, partial pivoting).  M and rhs are
// overwritten; returns false if (near-)singular.
static bool solve_dense(double* M, double* rhs, int n, double* x) {
  for (int col = 0; col < n; ++col) {
    int piv = col;
    double best = std::fabs(M[(size_t)col * n + col]);
    for (int r = col + 1; r < n; ++r) {
      double v = std::fabs(M[(size_t)r * n + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-300) return false;
    if (piv != col)
      for (int c = 0; c < n; ++c) {
        std::swap(M[(size_t)col * n + c], M[(size_t)piv * n + c]);
        if (c == 0) std::swap(rhs[col], rhs[piv]);
      }
    double diag = M[(size_t)col * n + col];
    for (int r = col + 1; r < n; ++r) {
      double f = M[(size_t)r * n + col] / diag;
      if (f != 0.0) {
        for (int c = col; c < n; ++c) M[(size_t)r * n + c] -= f * M[(size_t)col * n + c];
        rhs[r] -= f * rhs[col];
      }
    }
  }
  for (int r = n - 1; r >= 0; --r) {
    double s = rhs[r];
    for (int c = r + 1; c < n; ++c) s -= M[(size_t)r * n + c] * x[c];
    x[r] = s / M[(size_t)r * n + r];
  }
  return true;
}

// Primal active-set QP  min 1/2 d^T B d + g^T d  s.t.  lo<=d<=hi, A d >= a.
// Mirrors the torch reference (unit-norm rows, feasible start d=0, Bland's rule).
// Writes the step into d_out and [iters, converged] into info_out (length 2).
void qp_box_linear_f64(Float64Arr1D B, Float64Arr1D g, Float64Arr1D lo,
                       Float64Arr1D hi, Float64Arr1D A, Float64Arr1D a, int64_t P,
                       int64_t m, double tol, int64_t max_iter,
                       Float64MutArr1D d_out, Float64MutArr1D info_out) {
  const double* Bp = B.data();
  const double* gp = g.data();
  const double* lop = lo.data();
  const double* hip = hi.data();
  const double* Ap = (m > 0) ? A.data() : nullptr;
  const double* ap = (m > 0) ? a.data() : nullptr;
  double* d = d_out.data();
  double* info = info_out.data();
  nb::gil_scoped_release release;

  const int Pi = (int)P, mi = (int)m, mt = 2 * Pi + mi;
  std::vector<double> C((size_t)mt * Pi, 0.0), b(mt, 0.0);
  for (int i = 0; i < Pi; ++i) { C[(size_t)i * Pi + i] = 1.0; b[i] = lop[i]; }
  for (int i = 0; i < Pi; ++i) { C[(size_t)(Pi + i) * Pi + i] = -1.0; b[Pi + i] = -hip[i]; }
  for (int i = 0; i < mi; ++i) {
    for (int j = 0; j < Pi; ++j) C[(size_t)(2 * Pi + i) * Pi + j] = Ap[(size_t)i * Pi + j];
    b[2 * Pi + i] = ap[i];
  }
  for (int i = 0; i < mt; ++i) {  // unit-normalize rows
    double nrm = 0.0;
    for (int j = 0; j < Pi; ++j) { double v = C[(size_t)i * Pi + j]; nrm += v * v; }
    nrm = std::sqrt(nrm);
    if (nrm < 1e-300) nrm = 1e-300;
    for (int j = 0; j < Pi; ++j) C[(size_t)i * Pi + j] /= nrm;
    b[i] /= nrm;
  }
  for (int i = 0; i < Pi; ++i) d[i] = 0.0;
  std::vector<char> work(mt, 0);
  for (int i = 0; i < mt; ++i)
    if (std::fabs(b[i]) <= tol) work[i] = 1;  // active at d=0 (residual = -b)
  if (max_iter <= 0) max_iter = 10 * (Pi + mt) + 50;

  std::vector<double> cvec(Pi), p(Pi), kkt, rhs, sol;
  std::vector<int> idx;
  int iters = 0;
  for (iters = 1; iters <= max_iter; ++iters) {
    idx.clear();
    for (int i = 0; i < mt; ++i) if (work[i]) idx.push_back(i);
    const int k = (int)idx.size();
    for (int i = 0; i < Pi; ++i) {
      double s = gp[i];
      for (int j = 0; j < Pi; ++j) s += Bp[(size_t)i * Pi + j] * d[j];
      cvec[i] = s;
    }
    const int n = Pi + k;
    kkt.assign((size_t)n * n, 0.0);
    rhs.assign(n, 0.0);
    sol.assign(n, 0.0);
    for (int i = 0; i < Pi; ++i)
      for (int j = 0; j < Pi; ++j) kkt[(size_t)i * n + j] = Bp[(size_t)i * Pi + j];
    for (int r = 0; r < k; ++r) {
      const int row = idx[r];
      for (int j = 0; j < Pi; ++j) {
        const double cij = C[(size_t)row * Pi + j];
        kkt[(size_t)j * n + (Pi + r)] = -cij;   // -Cw^T (top-right)
        kkt[(size_t)(Pi + r) * n + j] = cij;     // Cw (bottom-left)
      }
    }
    for (int i = 0; i < Pi; ++i) rhs[i] = -cvec[i];
    solve_dense(kkt.data(), rhs.data(), n, sol.data());
    for (int i = 0; i < Pi; ++i) p[i] = sol[i];

    double pmax = 0.0;
    for (int i = 0; i < Pi; ++i) pmax = std::max(pmax, std::fabs(p[i]));
    if (pmax <= tol) {                    // step ~ 0: optimality / drop
      int jdrop = -1;
      for (int r = 0; r < k; ++r)
        if (sol[Pi + r] < -tol) { jdrop = r; break; }  // Bland: lowest-index
      if (k == 0 || jdrop < 0) { info[0] = (double)iters; info[1] = 1.0; return; }
      work[idx[jdrop]] = 0;
      continue;
    }
    double amin = 1.0;                     // ratio test (Bland tie-break)
    for (int i = 0; i < mt; ++i) {
      if (work[i]) continue;
      double cp = 0.0, cd = 0.0;
      for (int j = 0; j < Pi; ++j) {
        const double cij = C[(size_t)i * Pi + j];
        cp += cij * p[j];
        cd += cij * d[j];
      }
      if (cp < -tol) { double ratio = (b[i] - cd) / cp; if (ratio < amin) amin = ratio; }
    }
    int jblock = -1;
    if (amin < 1.0) {
      for (int i = 0; i < mt && jblock < 0; ++i) {
        if (work[i]) continue;
        double cp = 0.0, cd = 0.0;
        for (int j = 0; j < Pi; ++j) {
          const double cij = C[(size_t)i * Pi + j];
          cp += cij * p[j];
          cd += cij * d[j];
        }
        if (cp < -tol && (b[i] - cd) / cp <= amin + tol) jblock = i;
      }
    }
    const double alpha = (amin < 1.0) ? amin : 1.0;
    for (int i = 0; i < Pi; ++i) d[i] += alpha * p[i];
    if (jblock >= 0) work[jblock] = 1;
  }
  info[0] = (double)max_iter;
  info[1] = 0.0;
}

}  // namespace

NB_MODULE(_static_purc_core, m) {
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
  m.def("ipm_fast_backtrack_f64", &ipm_fast_backtrack_f64, "x"_a, "z"_a, "w"_a,
        "lam"_a, "dx"_a, "dz"_a, "dw"_a, "dlam"_a, "ap_max"_a, "ad_max"_a, "mu"_a,
        "done_in"_a, "rho"_a, "gamma_min"_a, "chi"_a, "ls_max"_a, "B"_a, "N"_a,
        "k"_a, "xf"_a, "zf"_a, "wf"_a, "lamf"_a, "done_out"_a,
        "Fused batched IPM fast-step backtracking (rho-decrease arm, GIL released).");
  m.def("ipm_safe_backtrack_f64", &ipm_safe_backtrack_f64, "x"_a, "z"_a, "w"_a,
        "lam"_a, "dxs"_a, "dzs"_a, "dws"_a, "dlams"_a, "ap_s"_a, "ad_s"_a, "mu"_a,
        "sigma_s"_a, "done_in"_a, "kappa"_a, "gamma_min"_a, "chi"_a, "ls_max"_a,
        "B"_a, "N"_a, "k"_a, "xs"_a, "zs"_a, "ws"_a, "lams"_a, "done_out"_a,
        "Fused batched IPM safe-step backtracking (Armijo arm, GIL released).");
  m.def("qp_box_linear_f64", &qp_box_linear_f64, "B"_a, "g"_a, "lo"_a, "hi"_a,
        "A"_a, "a"_a, "P"_a, "m"_a, "tol"_a, "max_iter"_a, "d_out"_a, "info_out"_a,
        "Primal active-set QP min 1/2 d'Bd + g'd s.t. lo<=d<=hi, A d>=a "
        "(unit-norm rows, Bland's rule, GIL released); writes d and [iters,conv].");
}
