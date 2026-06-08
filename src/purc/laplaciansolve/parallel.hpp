/**
 * A tiny persistent std::thread pool with a static-chunked parallel_for.
 *
 * The native core uses this instead of OpenMP so it links **no OpenMP runtime of
 * its own**.  That matters because CHOLMOD pulls in Homebrew's libomp and torch
 * pulls in its own; a third (ours) coexisting via KMP_DUPLICATE_LIB_OK crashes
 * once a threaded region actually spins up at large n.  With std::thread the core
 * is OpenMP-free and cross-platform (Linux/macOS/Windows) with no libomp link.
 *
 * Only embarrassingly-parallel batch loops use it (one independent system / RHS
 * per index), so a static contiguous partition with a fork-join barrier is all
 * that is needed - no work-stealing, no per-task allocation in the hot path.
 * The calling thread participates as one worker; the pool holds hardware-1
 * persistent workers that idle on a condition variable between calls (so there
 * is no per-call thread creation).  Workers never touch Python; callers release
 * the GIL around parallel_for.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace lsolve {

class ThreadPool {
 public:
  /// Process-wide pool (created lazily on first use; workers persist).
  static ThreadPool& instance() {
    static ThreadPool pool;
    return pool;
  }

  /// Hardware thread count (>= 1).
  static int hardware() {
    const int h = static_cast<int>(std::thread::hardware_concurrency());
    return h < 1 ? 1 : h;
  }

  /// Maximum parallelism used by parallel_for (>= 1).
  int num_threads() const { return n_threads_.load(std::memory_order_relaxed); }

  /// Set the parallelism cap, clamped to [1, hardware()].
  void set_num_threads(int n) {
    const int hw = hardware();
    if (n < 1) n = 1;
    if (n > hw) n = hw;
    n_threads_.store(n, std::memory_order_relaxed);
  }

  /**
   * Run body(i) for i in [0, n) across up to num_threads() workers.
   *
   * The range is split into contiguous static chunks.  Falls back to a serial
   * loop for n <= 1, a single thread, or a nested call from within a worker.
   */
  template <typename Body>
  void parallel_for(std::size_t n, Body body) {
    int p = num_threads();
    if (n <= 1 || p <= 1 || in_worker_) {
      for (std::size_t i = 0; i < n; ++i) body(i);
      return;
    }
    if (static_cast<std::size_t>(p) > n) p = static_cast<int>(n);

    auto run_chunk = [&](int t) {
      const std::size_t pp = static_cast<std::size_t>(p);
      const std::size_t begin = n * static_cast<std::size_t>(t) / pp;
      const std::size_t end = n * static_cast<std::size_t>(t + 1) / pp;
      for (std::size_t i = begin; i < end; ++i) body(i);
    };

    // All shared state is touched only under m_, so each worker sees a single
    // consistent generation (no cross-generation aliasing).  Worker chunks are
    // 1..p-1; the caller runs chunk 0 itself, then waits on the barrier.
    {
      std::unique_lock<std::mutex> lk(m_);
      chunk_fn_ = run_chunk;  // copy captures by-ref; valid until the barrier below
      njobs_ = p - 1;
      next_job_ = 1;
      pending_ = p - 1;
      ++gen_;
      cv_start_.notify_all();
    }
    run_chunk(0);
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_done_.wait(lk, [this] { return pending_ == 0; });
      chunk_fn_ = nullptr;
    }
  }

  /**
   * Run body(begin, end, worker_id) once per worker over a contiguous partition
   * of [0, n).
   *
   * Unlike parallel_for, the body receives its whole [begin, end) sub-range and
   * its worker id in [0, P), where P (<= num_threads()) is the number of chunks.
   * Each worker id is run by exactly one thread for the duration of the call (the
   * caller runs id 0; each pool worker claims at most one chunk), so the body may
   * index private per-thread state (e.g. its own CHOLMOD common/factor) by
   * worker_id with no locking.  Falls back to a single serial call body(0, n, 0)
   * for n <= 1, a single thread, or a nested call from within a worker.
   *
   * Shares the exact lock-based generation/barrier machinery of parallel_for
   * (the only difference is the chunk runner calls body once instead of looping),
   * so it inherits the same proven correctness.
   */
  template <typename Body>
  void parallel_for_ranges(std::size_t n, Body body) {
    int p = num_threads();
    if (n <= 1 || p <= 1 || in_worker_) {
      body(static_cast<std::size_t>(0), n, 0);
      return;
    }
    if (static_cast<std::size_t>(p) > n) p = static_cast<int>(n);

    auto run_chunk = [&](int t) {
      const std::size_t pp = static_cast<std::size_t>(p);
      const std::size_t begin = n * static_cast<std::size_t>(t) / pp;
      const std::size_t end = n * static_cast<std::size_t>(t + 1) / pp;
      body(begin, end, t);
    };

    {
      std::unique_lock<std::mutex> lk(m_);
      chunk_fn_ = run_chunk;
      njobs_ = p - 1;
      next_job_ = 1;
      pending_ = p - 1;
      ++gen_;
      cv_start_.notify_all();
    }
    run_chunk(0);
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_done_.wait(lk, [this] { return pending_ == 0; });
      chunk_fn_ = nullptr;
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

 private:
  ThreadPool() {
    n_threads_.store(hardware(), std::memory_order_relaxed);
    const int nw = hardware() - 1;  // the caller participates as one worker
    workers_.reserve(static_cast<std::size_t>(nw));
    for (int i = 0; i < nw; ++i) workers_.emplace_back([this] { worker_loop(); });
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lk(m_);
      stop_ = true;
      cv_start_.notify_all();
    }
    for (auto& t : workers_) {
      if (t.joinable()) t.join();
    }
  }

  void worker_loop() {
    in_worker_ = true;
    unsigned seen = 0;
    while (true) {
      int job = -1;
      std::function<void(int)> fn;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_start_.wait(lk, [this, seen] { return stop_ || gen_ != seen; });
        if (stop_) return;
        seen = gen_;
        if (next_job_ <= njobs_) {  // claim at most one chunk for this generation
          job = next_job_++;
          fn = chunk_fn_;
        }
      }
      if (job >= 0) {
        fn(job);
        std::unique_lock<std::mutex> lk(m_);
        if (--pending_ == 0) cv_done_.notify_one();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::function<void(int)> chunk_fn_;  // current generation's chunk runner
  int njobs_ = 0;                      // worker chunks this generation (1..njobs_)
  int next_job_ = 0;                   // next unclaimed chunk id
  int pending_ = 0;                    // chunks not yet finished
  unsigned gen_ = 0;                   // bumped each parallel_for
  bool stop_ = false;
  std::mutex m_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  std::atomic<int> n_threads_;
  static thread_local bool in_worker_;
};

inline thread_local bool ThreadPool::in_worker_ = false;

/// Run body(i) for i in [0, n) on the process-wide pool.
template <typename Body>
inline void parallel_for(std::size_t n, Body body) {
  ThreadPool::instance().parallel_for(n, std::move(body));
}

/// Run body(begin, end, worker_id) over a partition of [0, n) on the pool.
template <typename Body>
inline void parallel_for_ranges(std::size_t n, Body body) {
  ThreadPool::instance().parallel_for_ranges(n, std::move(body));
}

}  // namespace lsolve
