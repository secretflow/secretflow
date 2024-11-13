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

import abc
import ast
import glob
import importlib
import importlib.metadata
import logging
import os
import sys
from dataclasses import dataclass

from .resources import Resources


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


class Plugin(abc.ABC):
    '''
    Users can inherit from the Plugin for custom development and return a Plugin instance at the entry point.
    '''

    @abc.abstractmethod
    def get_resources(self) -> Resources: ...


@dataclass
class PluginEntry:
    def __init__(self, name: str, package: str, resources: Resources):
        if package == "":
            package = name
        self.name = name
        self.package = package
        self.resources = resources


_plugins: dict[str, PluginEntry] = {
    # buildin plugins
    "secretflow": PluginEntry("secretflow", "", Resources("component")),
}

PLUGIN_GROUP_NAME = "secretflow_plugins"


class PluginManager:
    @staticmethod
    def load():
        entry_points = importlib.metadata.entry_points().select(group=PLUGIN_GROUP_NAME)
        for ep in entry_points:
            if ep.name in _plugins:
                logging.error(f"ignore duplicate plugin, {ep.name} {ep.value}")
                continue

            entry_fn = ep.load()
            plugin = None
            if callable(entry_fn):
                plugin = entry_fn()
            else:
                logging.error(f"{ep.value} is not callable in plugin of {ep.name}")

            root_package = ep.module.split('.')[0]
            resource = plugin.get_resources() if isinstance(plugin, Plugin) else None
            if resource is None:
                resource = Resources.from_package(root_package)

            _plugins[ep.name] = PluginEntry(ep.name, root_package, resource)

    @staticmethod
    def get_plugins() -> dict[str, PluginEntry]:
        return _plugins

    @staticmethod
    def get_plugin(name: str) -> PluginEntry:
        assert name in _plugins, f"cannot find plugin<{name}>"
        return _plugins[name]


def load_plugins():
    PluginManager.load()
