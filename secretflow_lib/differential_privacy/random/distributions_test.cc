// Copyright (c) 2019 Ant Financial. All rights reserved.

#include "distributions.h"

#include <future>
#include <numeric>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "yacl/crypto/tools/prg.h"

namespace secretflow {
namespace dp {

template <typename T>
double get_mean(std::vector<T> const& v) {
  if (v.empty()) {
    return 0;
  }
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template <typename T>
double get_stdv(std::vector<T> const& v, float mean) {
  if (v.empty()) {
    return 0;
  }
  double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / v.size() - mean * mean);
  return stdev;
}

TEST(UniformRealTest, Test) {
  // GIVEN
  const float from = 0;
  const float to = 1;

  // WHEN
  std::random_device rd;
  yacl::Prg<uint64_t> prg(rd());
  UniformReal<float> uniform(from, to);

  std::vector<float> unif_real;
  for (int i = 0; i < 100; ++i) {
    unif_real.push_back(uniform(prg));
  }

  float mean = (from + to) / 2;
  float mean_unif_real = get_mean(unif_real);
  // THEN
  EXPECT_NEAR(mean_unif_real, mean, abs(mean) * .5);
}

TEST(BernoulliNegExpTest, Test) {
  // GIVEN
  const float p = 0.5;

  // WHEN
  std::random_device rd;
  yacl::Prg<uint64_t> prg(rd());
  BernoulliNegExp bernoulli(p);

  std::vector<float> bern;
  for (int i = 0; i < 100; ++i) {
    bern.push_back(bernoulli(prg));
  }

  float p_bern = get_mean(bern);

  // THEN
  EXPECT_NEAR(p_bern, p, p);
}

TEST(NormalRealTest, Test) {
  // GIVEN
  const float mean = 0;
  const float stdv = 1;

  // WHEN
  std::random_device rd;
  yacl::Prg<uint64_t> prg(rd());
  SecureNormalReal<double> secure_normal_real(mean, stdv);

  std::vector<float> norm_real;
  for (int i = 0; i < 100; ++i) {
    norm_real.push_back(secure_normal_real(prg));
  }

  float mean_norm_real = get_mean(norm_real);
  float stdv_norm_real = get_stdv(norm_real, mean_norm_real);

  // THEN
  EXPECT_NEAR(mean_norm_real, mean, stdv * .5);
  EXPECT_NEAR(stdv_norm_real, stdv, stdv * .5);
}

TEST(NormalDiscreteTest, Test) {
  // GIVEN
  const int mean = 1;
  const float stdv = 0.99;

  // WHEN
  std::random_device rd;
  yacl::Prg<uint64_t> prg(rd());
  NormalDiscrete<int> normal_discrete(mean, stdv);

  std::vector<float> norm_discrete;
  for (int i = 0; i < 50; ++i) {
    norm_discrete.push_back(normal_discrete(prg));
  }

  float mean_norm_discrete = get_mean(norm_discrete);
  float stdv_norm_discrete = get_stdv(norm_discrete, mean_norm_discrete);

  // THEN
  EXPECT_NEAR(mean_norm_discrete, mean, stdv);
  EXPECT_NEAR(stdv_norm_discrete, stdv, stdv);
}

TEST(LaplaceRealTest, Test) {
  // GIVEN
  const float mean = 0;
  const float stdv = 1;

  // WHEN
  std::random_device rd;
  yacl::Prg<uint64_t> prg(rd());
  SecureLaplaceReal<double> secure_laplace_real(mean, stdv);

  std::vector<float> lapl_real;
  for (int i = 0; i < 100; ++i) {
    lapl_real.push_back(secure_laplace_real(prg));
  }

  float mean_lapl_real = get_mean(lapl_real);
  float stdv_lapl_real = get_stdv(lapl_real, mean_lapl_real);

  // THEN
  EXPECT_NEAR(mean_lapl_real, mean, stdv);
  EXPECT_NEAR(stdv_lapl_real, stdv, stdv);
}

}  // namespace dp
}  // namespace secretflow