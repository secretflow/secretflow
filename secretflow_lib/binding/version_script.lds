VERS_1.0 {
  # Export symbols in pybind.
  global:
    PyInit_*;

  # Hide everything else.
  local:
    *;
};
