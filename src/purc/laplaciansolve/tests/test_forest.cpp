// Standalone C++ unit tests for the forest LDL factorization (run via ctest):
// the single-matrix solve residual, and the batched factorization's parity with
// looping the single-matrix solver - per-system (distinct values on one pattern,
// ragged RHS) and shared-matrix (one matrix, many RHS).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "forest_ldl.hpp"
#include "spmv.hpp"

using namespace lsolve;

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));              \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

// Relative residual ||M x - b|| / ||b|| for a full-CSC SDDM matrix.
static double rel_resid(const std::vector<int64_t>& indptr,
                        const std::vector<int64_t>& indices,
                        const std::vector<double>& data, int64_t n,
                        const double* x, const double* b) {
  std::vector<double> y(n);
  csc_matvec<int64_t, double>(indptr.data(), indices.data(), data.data(), n, n, x, y.data());
  double num = 0.0, den = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    num += (y[i] - b[i]) * (y[i] - b[i]);
    den += b[i] * b[i];
  }
  return std::sqrt(num / (den > 0 ? den : 1.0));
}

static double max_abs_diff(const double* a, const double* b, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

int main() {
  // Path-4 SDDM M = L_path + diag (a forest):
  //   diag [1.5, 2.5, 2.5, 1.5], off-diagonal -1 between neighbors.
  const std::vector<int64_t> indptr = {0, 2, 5, 8, 10};
  const std::vector<int64_t> indices = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
  const std::vector<double> data = {1.5, -1, -1, 2.5, -1, -1, 2.5, -1, -1, 1.5};
  const int64_t n = 4;
  const std::size_t nnz = data.size();

  // (1) single-matrix solve: residual must be ~0.
  ForestLDL<int64_t, double> f(indptr.data(), indices.data(), data.data(), n);
  CHECK(f.is_forest(), "path-4 must be detected as a forest");
  const std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> x(n);
  f.solve(b.data(), x.data(), 1);
  CHECK(rel_resid(indptr, indices, data, n, x.data(), b.data()) < 1e-12, "single solve residual");

  // (2) per-system batch: B matrices = data scaled per system (still forests),
  //     ragged RHS (2, 1, 3 right-hand sides). Compare to looping ForestLDL.
  ForestBatchSolver<int64_t, double> fb(indptr.data(), indices.data(), n);
  CHECK(fb.is_forest(), "batch path-4 must be a forest");
  const double scales[] = {1.0, 2.0, 0.5};
  const std::size_t B = 3;
  std::vector<double> values(B * nnz);
  for (std::size_t s = 0; s < B; ++s)
    for (std::size_t p = 0; p < nnz; ++p) values[s * nnz + p] = data[p] * scales[s];

  const std::vector<int64_t> offs = {0, 2, 3, 6};  // ragged: 2, 1, 3
  const std::size_t totalK = 6;
  std::vector<double> rhs(totalK * n), xout(totalK * n, 0.0);
  for (std::size_t r = 0; r < totalK; ++r)
    for (int64_t i = 0; i < n; ++i) rhs[r * n + i] = std::sin(0.7 * r + 1.3 * i) + 0.5;

  fb.solve_batch(values.data(), B, rhs.data(), offs.data(), xout.data());

  for (std::size_t s = 0; s < B; ++s) {
    std::vector<double> scaled(nnz);
    for (std::size_t p = 0; p < nnz; ++p) scaled[p] = data[p] * scales[s];
    ForestLDL<int64_t, double> fs(indptr.data(), indices.data(), scaled.data(), n);
    const int64_t off = offs[s];
    const int64_t k = offs[s + 1] - off;
    std::vector<double> xref(static_cast<std::size_t>(k * n));
    fs.solve(rhs.data() + off * n, xref.data(), static_cast<int64_t>(k));
    const double diff =
        max_abs_diff(xout.data() + off * n, xref.data(), static_cast<std::size_t>(k * n));
    CHECK(diff < 1e-12, "per-system batch matches looped single solve");
    // and residual against this system's own matrix
    CHECK(rel_resid(indptr, indices, scaled, n, xout.data() + off * n, rhs.data() + off * n) <
              1e-12,
          "per-system batch residual");
  }

  // (3) shared-matrix batch (one matrix, B RHS) vs looping the single solver.
  std::vector<double> rhsB(B * n), xsh(B * n, 0.0);
  for (std::size_t r = 0; r < B; ++r)
    for (int64_t i = 0; i < n; ++i) rhsB[r * n + i] = std::cos(0.9 * r - 0.4 * i);
  fb.solve_batch_shared(data.data(), B, rhsB.data(), xsh.data());
  for (std::size_t r = 0; r < B; ++r) {
    std::vector<double> xref(n);
    f.solve(rhsB.data() + r * n, xref.data(), 1);
    CHECK(max_abs_diff(xsh.data() + r * n, xref.data(), n) < 1e-12,
          "shared-matrix batch matches single solve");
  }

  if (g_failures == 0) {
    std::printf("All forest LDL tests passed.\n");
    return 0;
  }
  std::printf("%d forest LDL test(s) failed.\n", g_failures);
  return 1;
}
