// Self-contained uniform [0,1) sampler for the seeded production build.
//
// std::mt19937_64 is standardized (identical output across platforms), so a
// seed yields a reproducible factorization without threading a Python sample
// array through the bindings.  The bit-exact parity tests instead drive the
// build with an injected array stream; this generator is the convenience path.
#pragma once

#include <cstdint>
#include <random>

namespace lsolve {

class Mt64Uniform {
 public:
  explicit Mt64Uniform(std::uint64_t seed) : gen_(seed) {}

  // A uniform double in [0, 1) using the top 53 bits (full mantissa).
  double operator()() {
    return static_cast<double>(gen_() >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
  }

 private:
  std::mt19937_64 gen_;
};

}  // namespace lsolve
