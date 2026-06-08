// Bucketed approximate degree priority queue for the :deg elimination order.
//
// Faithful port of ApproxCholPQ / keyMap from Laplacians.jl
// (src/approxChol.jl), 0-based with -1 as the "no node" sentinel.  Vertices of
// (approximately) equal degree share a doubly-linked bucket; pop() returns a
// minimum-degree vertex, and inc()/dec() move a vertex only when its bucket
// index changes.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace lsolve {

template <typename Tind>
class ApproxCholPQ {
 public:
  static constexpr Tind kNone = static_cast<Tind>(-1);

  // Initialize from the per-vertex degrees.  `split` sizes the bucket array.
  ApproxCholPQ(const Tind* degs, std::size_t n, Tind split = 1)
      : n_(static_cast<Tind>(n)),
        split_(split),
        prev_(n, kNone),
        next_(n, kNone),
        key_(degs, degs + n),
        lists_(static_cast<std::size_t>(2 * split * static_cast<Tind>(n) + 2), kNone),
        minlist_(1) {
    for (std::size_t i = 0; i < n; ++i) {
      const Tind key = key_[i];
      const Tind head = lists_[key];  // raw key == key_map for the initial keys
      if (head >= 0) {
        next_[i] = head;
        prev_[head] = static_cast<Tind>(i);
      }
      lists_[key] = static_cast<Tind>(i);
    }
  }

  // Remove and return a vertex of minimum (approximate) degree.
  Tind pop() {
    while (lists_[minlist_] == kNone) {
      ++minlist_;
    }
    const Tind i = lists_[minlist_];
    const Tind nxt = next_[i];
    lists_[minlist_] = nxt;
    if (nxt >= 0) {
      prev_[nxt] = kNone;
    }
    return i;
  }

  // Decrement / increment the degree key of vertex i.
  void dec(Tind i) { adjust(i, static_cast<Tind>(-1)); }
  void inc(Tind i) { adjust(i, static_cast<Tind>(1)); }

 private:
  static Tind key_map(Tind x, Tind k, Tind upper) {
    return x <= k ? x : std::min(upper, k + x / k);
  }

  void move(Tind i, Tind newkey, Tind oldlist, Tind newlist) {
    const Tind prev = prev_[i];
    const Tind nxt = next_[i];
    if (nxt >= 0) {
      prev_[nxt] = prev;
    }
    if (prev >= 0) {
      next_[prev] = nxt;
    } else {
      lists_[oldlist] = nxt;
    }
    const Tind head = lists_[newlist];
    if (head >= 0) {
      prev_[head] = i;
    }
    lists_[newlist] = i;
    prev_[i] = kNone;
    next_[i] = head;
    key_[i] = newkey;
  }

  // Shared core of inc/dec: delta is +1 or -1.
  void adjust(Tind i, Tind delta) {
    const Tind k = split_ * n_;
    const Tind upper = 2 * split_ * n_ + 1;
    const Tind oldkey = key_[i];
    const Tind oldlist = key_map(oldkey, k, upper);
    const Tind newlist = key_map(oldkey + delta, k, upper);
    if (newlist != oldlist) {
      move(i, oldkey + delta, oldlist, newlist);
      if (delta < 0 && newlist < minlist_) {
        minlist_ = newlist;
      }
    } else {
      key_[i] = oldkey + delta;
    }
  }

  Tind n_;
  Tind split_;
  std::vector<Tind> prev_;
  std::vector<Tind> next_;
  std::vector<Tind> key_;
  std::vector<Tind> lists_;
  Tind minlist_;
};

}  // namespace lsolve
