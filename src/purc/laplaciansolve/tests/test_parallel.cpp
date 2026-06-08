// Standalone C++ unit tests for the std::thread pool's parallel_for_ranges
// (run via ctest).  Verifies the worker-indexed partition that the batched
// CHOLMOD factorization relies on: the chunks tile [0, n) exactly once with no
// gap/overlap, and each worker id is handed to exactly one chunk per call.

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "parallel.hpp"

using namespace lsolve;

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg));              \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

// Run parallel_for_ranges at the given thread cap and check the partition.
static void check_tiling(std::size_t n, int cap) {
  ThreadPool::instance().set_num_threads(cap);

  // Disjoint chunks by construction, so each cover[i] is written by exactly one
  // thread; a value != 1 afterwards means a gap (formula bug), never a race.
  std::vector<int> cover(n, 0);
  std::mutex mu;
  std::set<int> ids;
  std::vector<std::pair<std::size_t, std::size_t>> ranges;

  parallel_for_ranges(n, [&](std::size_t begin, std::size_t end, int w) {
    for (std::size_t i = begin; i < end; ++i) cover[i] += 1;
    std::lock_guard<std::mutex> lk(mu);
    ids.insert(w);
    ranges.push_back({begin, end});
  });

  char ctx[64];
  std::snprintf(ctx, sizeof(ctx), "n=%zu cap=%d", n, cap);

  for (std::size_t i = 0; i < n; ++i) {
    if (cover[i] != 1) {
      std::printf("FAIL %s: index %zu covered %d times\n", ctx, i, cover[i]);
      ++g_failures;
      break;
    }
  }
  // Every chunk has a distinct worker id (no two chunks share private state).
  CHECK(ids.size() == ranges.size(), ctx);
  // Worker ids are a contiguous [0, #chunks) range.
  for (int w : ids) CHECK(w >= 0 && w < static_cast<int>(ranges.size()), ctx);
  // At least one chunk always runs (covers the n == 0 empty-range case too).
  CHECK(!ranges.empty(), ctx);
}

int main() {
  const int hw = ThreadPool::hardware();
  const std::size_t sizes[] = {0, 1, 2, 3, 7, 15, 16, 17, 1000, 100003};
  const int caps[] = {1, 2, 3, 4, 8, hw};

  for (std::size_t n : sizes) {
    for (int c : caps) {
      check_tiling(n, c < 1 ? 1 : c);
    }
  }

  ThreadPool::instance().set_num_threads(hw);  // restore default

  if (g_failures == 0) {
    std::printf("All parallel_for_ranges tests passed.\n");
    return 0;
  }
  std::printf("%d parallel_for_ranges test(s) failed.\n", g_failures);
  return 1;
}
