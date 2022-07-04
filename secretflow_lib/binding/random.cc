#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <optional>

#include "binding.h"
#include "secretflow_lib/differential_privacy/random/distributions.h"

namespace py = pybind11;

template <typename T, typename Dist>
py::handle distribution(std::optional<std::vector<size_t>> size, Dist &dist) {
  std::vector<size_t> shape = size ? size.value() : std::vector<size_t>({});

  size_t nelems = 1;
  for (auto s : shape) {
    nelems *= s;
  }
  T *buffer = new T[nelems];

  std::random_device rd;
  yasl::PseudoRandomGenerator<uint64_t> prg(rd());
  for (size_t i = 0; i < nelems; ++i) {
    buffer[i] = dist(prg);
  }

  constexpr size_t elem_size = sizeof(T);
  size_t ndim = shape.size();
  std::vector<size_t> strides(ndim);
  if (ndim > 0) {
    strides[ndim - 1] = elem_size;
  }
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  // NOTE: bind raw buffer lifetime with numpy array.
  py::capsule base(buffer, [](void *o) { delete[] static_cast<T *>(o); });
  py::array a(shape, strides, buffer, base);

  return a.release();
}

py::handle uniform_real(float low, float high,
                        std::optional<std::vector<size_t>> size) {
  secretflow::dp::UniformReal<float> dist(low, high);
  return distribution<float>(size, dist);
}

py::handle bernoulli_neg_exp(float p, std::optional<std::vector<size_t>> size) {
  secretflow::dp::BernoulliNegExp dist(p);
  return distribution<float>(size, dist);
}

py::handle secure_normal_real(float mean, float stdv,
                              std::optional<std::vector<size_t>> size) {
  secretflow::dp::SecureNormalReal<float> dist(mean, stdv);
  return distribution<float>(size, dist);
}

py::handle normal_discrete(float mean, float stdv,
                           std::optional<std::vector<size_t>> size) {
  secretflow::dp::NormalDiscrete<int> dist(mean, stdv);
  return distribution<int>(size, dist);
}

py::handle secure_laplace_real(float mean, float stdv,
                               std::optional<std::vector<size_t>> size) {
  secretflow::dp::SecureLaplaceReal<float> dist(mean, stdv);
  return distribution<float>(size, dist);
}

void module_random(pybind11::module &m) {
  m.doc() = "Secure Random Number Generation";

  m.def("uniform_real", &uniform_real, py::arg("low") = 0.0,
        py::arg("high") = 1.0, py::arg("size").none(true) = std::nullopt);

  m.def("bernoulli_neg_exp", &bernoulli_neg_exp, py::arg("p") = 0.5,
        py::arg("size").none(true) = std::nullopt);

  m.def("secure_normal_real", &secure_normal_real, py::arg("mean") = 0.0,
        py::arg("stdv") = 1.0, py::arg("size").none(true) = std::nullopt);

  m.def("normal_discrete", &normal_discrete, py::arg("mean") = 0.0,
        py::arg("stdv") = 1.0, py::arg("size").none(true) = std::nullopt);

  m.def("secure_laplace_real", &secure_laplace_real, py::arg("mean") = 0.0,
        py::arg("stdv") = 1.0, py::arg("size").none(true) = std::nullopt);
}