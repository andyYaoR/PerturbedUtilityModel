// Standalone C++ unit tests for the LaplacianSolve core (run via ctest).
//
// Uses a hand-built factorization of the 3-node path Laplacian
//   L = [[1,-1,0],[-1,2,-1],[0,-1,1]]
// whose mean-zero solution for b = [1,-2,1] is [1/3,-2/3,1/3] (computed by
// hand), so these checks need no external reference data.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "ldl_solve.hpp"
#include "pcg.hpp"
#include "spmv.hpp"

using namespace lsolve;

static int g_failures = 0;

#define CHECK_CLOSE(a, b, tol)                                                 \
  do {                                                                         \
    const double _a = (a), _b = (b);                                           \
    if (std::abs(_a - _b) > (tol)) {                                           \
      std::printf("FAIL %s:%d  %s=%.8g vs %s=%.8g (tol %.1e)\n", __FILE__,     \
                  __LINE__, #a, _a, #b, _b, static_cast<double>(tol));         \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

int main() {
  // LDLinv of the path Laplacian via leaf elimination (exact for a tree).
  const std::vector<int64_t> col = {0, 1};
  const std::vector<int64_t> colptr = {0, 1, 2};
  const std::vector<int64_t> rowval = {1, 2};
  const std::vector<double> fval = {1.0, 1.0};
  const std::vector<double> d = {1.0, 1.0, 0.0};
  const std::vector<double> b = {1.0, -2.0, 1.0};

  std::vector<double> x(3);
  ldl_solve<int64_t, double>(col.data(), colptr.data(), rowval.data(),
                             fval.data(), d.data(), 2, 3, b.data(), x.data(),
                             /*subtract_mean=*/true);
  CHECK_CLOSE(x[0], 1.0 / 3.0, 1e-12);
  CHECK_CLOSE(x[1], -2.0 / 3.0, 1e-12);
  CHECK_CLOSE(x[2], 1.0 / 3.0, 1e-12);

  // SpMV on the CSC Laplacian must recover b from the solution.
  const std::vector<int64_t> indptr = {0, 2, 5, 7};
  const std::vector<int64_t> indices = {0, 1, 0, 1, 2, 1, 2};
  const std::vector<double> data = {1, -1, -1, 2, -1, -1, 1};
  std::vector<double> y(3);
  csc_matvec<int64_t, double>(indptr.data(), indices.data(), data.data(), 3, 3,
                              x.data(), y.data());
  CHECK_CLOSE(y[0], 1.0, 1e-12);
  CHECK_CLOSE(y[1], -2.0, 1e-12);
  CHECK_CLOSE(y[2], 1.0, 1e-12);

  // CG (identity preconditioner) recovers the mean-zero solution.
  std::vector<double> xpcg(3, 0.0);
  auto matvec = [&](const double* in, double* out) {
    csc_matvec<int64_t, double>(indptr.data(), indices.data(), data.data(), 3,
                                3, in, out);
  };
  auto identity = [&](const double* in, double* out) {
    for (int i = 0; i < 3; ++i) out[i] = in[i];
  };
  const auto res =
      pcg<double>(3, b.data(), matvec, identity, 1e-12, 100, 0, xpcg.data());
  CHECK_CLOSE(xpcg[0], 1.0 / 3.0, 1e-8);
  CHECK_CLOSE(xpcg[1], -2.0 / 3.0, 1e-8);
  CHECK_CLOSE(xpcg[2], 1.0 / 3.0, 1e-8);
  if (!res.converged) {
    std::printf("FAIL: CG did not converge (relres=%.3e)\n", res.relres);
    ++g_failures;
  }

  if (g_failures == 0) {
    std::printf("All C++ core tests passed.\n");
    return 0;
  }
  std::printf("%d C++ core test(s) failed.\n", g_failures);
  return 1;
}
