// Approximate-Cholesky elimination (the adaptive :deg build).
//
// Faithful port of approxChol(a::LLmatp) from Laplacians.jl
// (src/approxChol.jl): eliminate the lowest-degree vertex, and for each
// non-final edge sample one survivor edge weighted by the column's cumulative
// mass, recording the elimination multiplier.  Randomness is supplied as an
// explicit sample stream (a pointer to uniform [0,1) doubles consumed in order)
// so the build is reproducible and bit-exactly checkable against the reference.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "approx_chol_pq.hpp"
#include "linked_list_matrix.hpp"
#include "rng.hpp"

namespace lsolve {

// The approxChol factorization in the array layout consumed by ldl_solve.
template <typename Tind, typename Tval>
struct LdlFactor {
  std::vector<Tind> col;     // elimination order (length n-1)
  std::vector<Tind> colptr;  // column pointers (length n)
  std::vector<Tind> rowval;  // L row indices
  std::vector<Tval> fval;    // L multipliers
  std::vector<Tval> d;       // diagonal (length n)
};

// Sampler that replays a fixed array of uniform draws (throws if exhausted).
template <typename Tval>
struct ArraySampler {
  const Tval* samples;
  std::size_t n;
  std::size_t i = 0;
  Tval operator()() {
    if (i >= n) {
      throw std::runtime_error(
          "approx_chol: sample stream exhausted; supply more draws.");
    }
    return samples[i++];
  }
};

// Eliminate `a` (mutated in place); `sample()` returns uniform draws in [0,1).
template <typename Tind, typename Tval, typename Sampler>
LdlFactor<Tind, Tval> approx_chol_deg(LLmatp<Tind, Tval>& a, Sampler&& sample) {
  const Tind n = a.n;
  LdlFactor<Tind, Tval> f;
  f.col.assign(n > 0 ? static_cast<std::size_t>(n - 1) : 0, 0);
  f.colptr.assign(static_cast<std::size_t>(n), 0);
  f.d.assign(static_cast<std::size_t>(n), Tval(0));

  ApproxCholPQ<Tind> pq(a.degs.data(), static_cast<std::size_t>(n));

  std::vector<Tind> colspace;
  std::vector<Tind> compressed;
  std::vector<Tval> cumspace;

  Tind it = 0;
  while (it < n - 1) {
    const Tind i = pq.pop();
    f.col[it] = i;
    f.colptr[it] = static_cast<Tind>(f.rowval.size());
    ++it;

    get_ll_col(a, i, colspace);
    compress_col(a, colspace, pq, compressed);
    const Tind length = static_cast<Tind>(compressed.size());

    cumspace.resize(static_cast<std::size_t>(length));
    Tval csum = Tval(0);
    for (Tind ii = 0; ii < length; ++ii) {
      csum += a.val[compressed[ii]];
      cumspace[ii] = csum;
    }
    Tval wdeg = csum;
    Tval col_scale = Tval(1);

    for (Tind joffset = 0; joffset < length - 1; ++joffset) {
      const Tind ll = compressed[joffset];
      const Tval w = a.val[ll] * col_scale;
      const Tind j = a.row[ll];
      const Tind revj = a.reverse[ll];
      const Tval fmul = w / wdeg;

      const Tval r = sample() * (csum - cumspace[joffset]) + cumspace[joffset];
      Tind koff = static_cast<Tind>(
          std::lower_bound(cumspace.begin(), cumspace.begin() + length, r) -
          cumspace.begin());
      if (koff >= length) {
        koff = length - 1;
      }
      const Tind k = a.row[compressed[koff]];

      pq.inc(k);
      const Tval new_edge_val = fmul * (Tval(1) - fmul) * wdeg;

      // Reuse revj as edge (j -> k) in column j; move ll into column k as (k -> j).
      a.row[revj] = k;
      a.val[revj] = new_edge_val;
      a.reverse[revj] = ll;

      const Tind khead = a.cols[k];
      a.cols[k] = ll;
      a.next[ll] = khead;
      a.reverse[ll] = revj;
      a.val[ll] = new_edge_val;
      a.row[ll] = j;

      col_scale *= (Tval(1) - fmul);
      wdeg *= (Tval(1) - fmul) * (Tval(1) - fmul);

      f.rowval.push_back(j);
      f.fval.push_back(fmul);
    }

    // Final edge carries the pivot weight into d[i].
    const Tind ll = compressed[static_cast<std::size_t>(length - 1)];
    const Tval w = a.val[ll] * col_scale;
    const Tind j = a.row[ll];
    const Tind revj = a.reverse[ll];

    if (it < n - 1) {
      pq.dec(j);
    }
    a.val[revj] = Tval(0);

    f.rowval.push_back(j);
    f.fval.push_back(Tval(1));
    f.d[i] = w;
  }

  f.colptr[static_cast<std::size_t>(it)] = static_cast<Tind>(f.rowval.size());
  return f;
}

// Convenience: build from a CSC adjacency + a replayed sample array.
template <typename Tind, typename Tval>
LdlFactor<Tind, Tval> approx_chol(const Tind* indptr, const Tind* indices,
                                  const Tval* data, std::size_t n,
                                  const Tval* samples, std::size_t nsamples) {
  LLmatp<Tind, Tval> a = build_llmatp<Tind, Tval>(indptr, indices, data, n);
  ArraySampler<Tval> sampler{samples, nsamples, 0};
  return approx_chol_deg<Tind, Tval>(a, sampler);
}

// Convenience: build from a CSC adjacency using a seeded internal RNG.
template <typename Tind, typename Tval>
LdlFactor<Tind, Tval> approx_chol_seeded(const Tind* indptr, const Tind* indices,
                                         const Tval* data, std::size_t n,
                                         std::uint64_t seed) {
  LLmatp<Tind, Tval> a = build_llmatp<Tind, Tval>(indptr, indices, data, n);
  Mt64Uniform rng(seed);
  return approx_chol_deg<Tind, Tval>(a, rng);
}

}  // namespace lsolve
