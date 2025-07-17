# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import sys

import nbformat

# tool to clean notebook for ci check forbidden {{}} and {# #}


def forbidden_found(output):
    pattern = re.compile(r'(\{\{|\{#)')
    for key in ['text', 'evalue']:
        if key in output and pattern.search(str(output[key])):
            return True
    if 'data' in output:
        for val in output['data'].values():
            if pattern.search(str(val)):
                return True
    return False


def clean_notebook(filename):
    nb = nbformat.read(filename, as_version=nbformat.NO_CONVERT)
    changed = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'outputs' in cell:
            outputs = cell.outputs
            if any(forbidden_found(o) for o in outputs):
                # Remove all outputs and reset execution_count
                cell.outputs = []
                changed = True
    if changed:
        nbformat.write(nb, filename)
        print(f"Cleaned forbidden outputs and execution_counts in {filename}")
    else:
        print(f"No forbidden outputs found in {filename}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python clean_forbidden_outputs.py notebook.ipynb")
        sys.exit(1)
    clean_notebook(sys.argv[1])
