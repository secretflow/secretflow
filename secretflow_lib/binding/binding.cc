#include "binding.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_lib, m) {
  m.doc() = "Yet Another Simple library";
  py::module m_random = m.def_submodule("random");
  module_random(m_random);
}