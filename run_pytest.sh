#!/bin/bash
coverage erase
unset JOB_NAME & env PYTHONPATH=$PYTHONPATH:bazel-bin python -m coverage run -p tests/main.py "$@"
coverage combine
coverage report -m > tests/result.md
echo "cat tests/result.md"
cat tests/result.md
