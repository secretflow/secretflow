#pragma once

#include <math.h>

#include <limits>
#include <random>
#include <type_traits>
#include <typeinfo>

#include "yacl/base/exception.h"
#include "yacl/crypto/tools/prg.h"

namespace secretflow {
namespace dp {

namespace {
template <class T>
struct is_normal_descrete_support_type : public std::false_type {};

template <>
struct is_normal_descrete_support_type<int> : public std::true_type {};

// template <>
// struct is_normal_descrete_support_type<uint64_t> : public std::true_type {};

// template <>
// struct is_normal_descrete_support_type<uint128_t> : public std::true_type {};
}  // namespace

/**
 * Samples a uniform distribution in the range [from, to) of type T.
 * @param prg pseudo random number generator
 * return: a sample of real numbers from a uniform distribution
 */

template <typename T>
struct UniformReal {
  inline UniformReal(T from, T to) {
    YACL_ENFORCE(from <= to);
    YACL_ENFORCE(to - from <= std::numeric_limits<T>::max());
    from_ = from;
    to_ = to;
  }

  template <typename V>
  T operator()(yacl::crypto::Prg<V> &prg) {
    constexpr auto MASK = static_cast<V>(
        (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
    constexpr auto DIVISOR =
        static_cast<T>(1) /
        (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
    T x = (prg() & MASK) * DIVISOR;
    return (x * (to_ - from_) + from_);
  }

 private:
  T from_;
  T to_;
};

/**

 * Sample from Bernoulli(exp(-gamma)).

 * @param gamma param to sample from Bernoulli(exp(-gamma)), must be
 non-negative
 * @param prg pseudo random number generator
 * return: a sample from the Bernoulli(exp(-gamma)) distribution

 * [CK20] Canonne, Kamath, Steinke, "The Discrete Gaussian for Differential
 Privacy"
 */

struct BernoulliNegExp {
  inline BernoulliNegExp(double gamma) {
    // Gamma must be non-negative
    YACL_ENFORCE(gamma >= 0);
    gamma_ = gamma;
  }

  template <typename V>
  int operator()(yacl::crypto::Prg<V> &prg) {
    while (gamma_ > 1) {
      gamma_ -= 1;

      struct BernoulliNegExp bernoulli(1);
      if (!bernoulli(prg)) {
        return 0;
      }
    }

    UniformReal<double> uniform(0.0, 1.0);
    int counter = 1;
    while (uniform(prg) <= gamma_ / counter) {
      counter++;
    }

    return counter % 2;
  }

 private:
  double gamma_;
};

/**
 * Samples a normal distribution using the Box-Muller method.

 * Samples from the Gaussian distribution are generated using two samples from
 * normal distribution using the Box-Muller method as detailed in [HB21b],
 * to prevent against reconstruction attacks due to limited floating point
 precision.

 * @param prg pseudo random number generator
 * return: a sample of real numbers from the normal distribution

 * [HB21] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in
 Differential Privacy." arXiv preprint arXiv:2107.10138 (2021).
 */

template <typename T>
struct SecureNormalReal {
  inline SecureNormalReal(T mean, T stdv) {
    YACL_ENFORCE(stdv >= 0);
    mean_ = mean;
    stdv_ = stdv;
  }

  inline T transform(T val, T mean, T std) { return val * std + mean; }

  inline constexpr T pi() { return static_cast<T>(3.14159265358979323846L); }

  template <typename V>
  T operator()(yacl::crypto::Prg<V> &prg) {
    UniformReal<T> uniform(0.0, 1.0);
    const T u1 = uniform(prg);
    const T u2 = uniform(prg);
    const T r =
        std::sqrt(static_cast<T>(-2.0) * std::log(static_cast<T>(1.0) - u2));
    const T theta = static_cast<T>(2.0) * pi() * u1;

    T n1, n2;
    n1 = transform(r * std::sin(theta), mean_, stdv_);
    n2 = transform(r * std::cos(theta), mean_, stdv_);
    return (n1 + n2) / std::sqrt(static_cast<T>(2.0));
  }

 private:
  T mean_;
  T stdv_;
};

/**
 * The Discrete Gaussian mechanism in differential privacy
 * Re-purposed for approximate :math:`(\epsilon,\delta)`-differential privacy.

  * @param prg pseudo random number generator
  * return: a sample of integers from the discrete normal distribution

 * [CK20] Canonne, Kamath, Steinke, "The Discrete Gaussian for Differential
 Privacy"
 */

template <typename T, typename Enable = void>
struct NormalDiscrete {};

template <typename T>
struct NormalDiscrete<T, typename std::enable_if<
                             is_normal_descrete_support_type<T>::value>::type> {
  inline NormalDiscrete(T mean, double stdv) {
    YACL_ENFORCE(stdv >= 0);
    mean_ = mean;
    stdv_ = stdv;
  }

  template <typename V>
  T operator()(yacl::crypto::Prg<V> &prg) {
    if (stdv_ == 0) {
      return mean_;
    }

    double tau = 1 / (1 + floor(stdv_));
    double sigma2 = pow(stdv_, 2);
    BernoulliNegExp bernoulli1(tau);

    std::binomial_distribution<> binomial(1, 0.5);
    // generate a new generator for std::binomial_distribution
    // TODO: Adapt the official PRG interface of cpp11 for Prg
    yacl::crypto::Prg<uint128_t> rd(0, yacl::crypto::PRG_MODE::kAesEcb);
    std::mt19937 gen(rd());

    while (true) {
      int geom_x = 0;
      while (bernoulli1(prg)) {
        geom_x += 1;
      }

      int bern_b = binomial(gen);
      if (bern_b & (!geom_x)) {
        continue;
      }

      T lap_y = static_cast<T>((1 - 2 * bern_b) * geom_x);

      BernoulliNegExp bernoulli(pow((abs(lap_y) - tau * sigma2), 2) / 2 /
                                sigma2);
      if (bernoulli(prg)) {
        return stdv_ + lap_y;
      }
    }
  }

 private:
  T mean_;
  double stdv_;
};

/**
 * The classical Laplace mechanism in differential privacy.

 * Samples are generated using 4 uniform variates, as detailed in [HB21]_, to
 prevent
 * against reconstruction attacks due to limited floating point precision.

 * @param prg pseudo random number generator
 * return: a sample of real numbers from the Laplace distribution

 * [HB21] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in
 Differential Privacy." arXiv preprint arXiv:2107.10138 (2021).
 */

template <typename T>
struct SecureLaplaceReal {
  inline SecureLaplaceReal(T mean, T stdv) {
    YACL_ENFORCE(stdv >= 0);
    mean_ = mean;
    stdv_ = stdv;
  }

  inline T transform(T val, T mean, T std) { return val * std + mean; }

  inline constexpr T pi() { return static_cast<T>(3.14159265358979323846L); }

  template <typename V>
  T operator()(yacl::crypto::Prg<V> &prg) {
    UniformReal<T> uniform(0.0, 1.0);
    const T u1 = uniform(prg);
    const T u2 = uniform(prg);
    const T u3 = uniform(prg);
    const T u4 = uniform(prg);

    T la;
    la = std::log(static_cast<T>(1.0) - u1) * std::cos(pi() * u2) +
         std::log(static_cast<T>(1.0) - u3) * std::cos(pi() * u4);
    return transform(la, mean_, stdv_);
  }

 private:
  T mean_;
  T stdv_;
};

}  // namespace dp
}  // namespace secretflow