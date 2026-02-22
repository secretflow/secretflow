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
import importlib
import importlib.metadata
import logging
from dataclasses import dataclass

from .resources import Resources


class Plugin(abc.ABC):
    """
    Users can inherit from the Plugin for custom development and return a Plugin instance at the entry point.
    """

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

            root_package = ep.module.split(".")[0]
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
