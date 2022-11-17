#!/bin/bash

set -ex
coverage erase
unset JOB_NAME & env PYTHONPATH=$PYTHONPATH:bazel-bin DGLBACKEND=tensorflow python -m coverage run -p tests/main.py "$@"
coverage combine
coverage report -m > tests/result.md
coverage xml -o coverage.xml
echo "cat tests/result.md"
cat tests/result.md
