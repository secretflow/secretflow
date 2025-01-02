# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import glob
import importlib
import importlib.metadata
import logging
import os
import sys


def _check_module_usage(file_path: str) -> bool:
    module_name = "secretflow.component.core"
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    has_parse_import = False
    tree = ast.parse(file_content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            has_parse_import = True
            for alias in node.names:
                if alias.name.startswith(module_name):
                    return True
        elif isinstance(node, ast.ImportFrom):
            has_parse_import = True
            if node.module.startswith(module_name):
                return True
        elif has_parse_import:
            # early stop, just parse import of file header.
            return False

    return False


def load_component_modules(
    root_path: str,
    module_prefix: str = "",
    ignore_dirs: list[str] = [],
    ignore_keys: list[str] = [],
    ignore_root_files: bool = True,
):
    if root_path not in sys.path:
        sys.path.append(root_path)

    def is_ignore_file(file):
        for key in ignore_keys:
            if key in file:
                return True
        return False

    if ignore_root_files:
        root_dirs = [
            f
            for f in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, f))
        ]
        root_dirs = [x for x in root_dirs if x not in ignore_dirs]
    else:
        root_dirs = [root_path]

    for dir_name in root_dirs:
        if dir_name.startswith("__"):  # ignore __pycache__
            continue
        pattern = os.path.join(root_path, dir_name, "**/*.py")
        for pyfile in glob.glob(pattern, recursive=True):
            if pyfile.endswith("__init__.py") or is_ignore_file(pyfile):
                continue
            if not _check_module_usage(pyfile):
                continue

            module_name = (
                os.path.relpath(pyfile, root_path)
                .removesuffix(".py")
                .replace(os.path.sep, ".")
            )
            if module_prefix:
                module_name = f"{module_prefix}.{module_name}"

            try:
                importlib.import_module(module_name)
            except Exception as e:
                raise ValueError(
                    f"import component fail, file={pyfile}, module={module_name}, err={e}"
                )
