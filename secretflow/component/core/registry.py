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


from secretflow.error_system.exceptions import NotSupportedError
from secretflow.spec.v1.component_pb2 import ComponentDef

from .component import Component
from .definition import Definition


class Registry:
    _definitions: dict[str, Definition] = {}
    _class_map: dict[str, Definition] = {}

    @staticmethod
    def register(d: Definition):
        key = Registry.gen_key(d.domain, d.name, d.version)
        if key in Registry._definitions:
            raise ValueError(f"{key} is already registered")
        Registry._definitions[key] = d
        Registry._class_map[d.class_id] = d

    @staticmethod
    def unregister(domain: str, name: str, version: str) -> bool:
        key = Registry.gen_key(domain, name, version)
        if key not in Registry._definitions:
            return False
        d = Registry._definitions.pop(key)
        del Registry._class_map[d.class_id]
        return True

    @staticmethod
    def get_definition(domain: str, name: str, version: str) -> Definition:
        key = Registry.gen_key(domain, name, version)
        return Registry._definitions.get(key)

    @staticmethod
    def get_definition_by_class(cls: Component | type[Component]) -> Definition:
        if isinstance(cls, Component):
            cls = type(cls)

        cls_id = Definition.to_class_id(cls)
        return Registry._class_map.get(cls_id)

    @staticmethod
    def get_component_defs() -> list[ComponentDef]:  # type: ignore
        result = []
        for d in Registry._definitions.values():
            result.append(d.component_def)
        return result

    @staticmethod
    def gen_key(domain: str, name: str, version: str) -> str:
        tokens = version.split('.')
        if len(tokens) != 3:
            raise NotSupportedError.not_supported_version(
                f"Registry version must be in format of x.y.z, but got {version}"
            )
        major = tokens[0]
        return f"{domain}/{name}:{major}"


def register(domain: str, version: str, name: str = "", desc: str = None):
    if domain == "" or version == "":
        raise NotSupportedError.not_supported_version(
            f"In register, domain<{domain}> and version<{version}> cannot be empty"
        )

    def wrapper(cls):
        d = Definition(cls, domain, version, name, desc)
        Registry.register(d)
        return cls

    return wrapper
