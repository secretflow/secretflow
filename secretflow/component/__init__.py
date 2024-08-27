# Copyright 2023 Ant Group Co., Ltd.
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


import glob
import importlib
import os


def _import_all_submodules():
    ignore_dirs = ["core", "test_framework", "storage"]
    ignore_keys = ["/core/", "_utils/", "/ml/nn/"]

    def is_ignore_file(file):
        for key in ignore_keys:
            if key in file:
                return True
        return False

    root_path = os.path.dirname(__file__)
    root_dirs = [
        f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))
    ]
    root_dirs = [x for x in root_dirs if x not in ignore_dirs]

    for dir_name in root_dirs:
        if dir_name.startswith("__"):  # ignore __pycache__
            continue
        pattern = os.path.join(root_path, dir_name, "**/*.py")
        for pyfile in glob.glob(pattern, recursive=True):
            if pyfile.endswith("__init__.py") or is_ignore_file(pyfile):
                continue
            pyfile = pyfile.removeprefix(root_path)
            pyfile = pyfile.removesuffix(".py")
            module_name = "secretflow.component" + ".".join(pyfile.split("/"))
            importlib.import_module(module_name)


_import_all_submodules()
