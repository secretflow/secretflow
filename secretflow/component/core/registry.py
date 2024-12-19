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


from collections import defaultdict
from typing import Iterable

from secretflow.error_system.exceptions import NotSupportedError
from secretflow.spec.v1.component_pb2 import CompListDef, ComponentDef

from .component import Component
from .definition import Definition

COMP_LIST_NAME = "secretflow"
COMP_LIST_DESC = "First-party SecretFlow components."
COMP_LIST_VERSION = "0.0.1"

_reg_defs_by_key: dict[str, Definition] = {}
_reg_defs_by_cls: dict[str, Definition] = {}
_reg_defs_by_pkg: dict[str, list[Definition]] = defaultdict(list)
_comp_list_def = CompListDef(
    name=COMP_LIST_NAME, desc=COMP_LIST_DESC, version=COMP_LIST_VERSION
)


def _gen_reg_key(domain: str, name: str, version: str) -> str:
    tokens = version.split('.')
    if len(tokens) != 3:
        raise NotSupportedError.not_supported_version(
            f"Registry version must be in format of x.y.z, but got {version}"
        )
    major = tokens[0]
    return f"{domain}/{name}:{major}"


class Registry:
    @staticmethod
    def register(d: Definition):
        key = _gen_reg_key(d.domain, d.name, d.version)
        if key in _reg_defs_by_key:
            raise ValueError(f"{key} is already registered")
        _reg_defs_by_key[key] = d
        _reg_defs_by_cls[d.class_id] = d
        _reg_defs_by_pkg[d.root_package].append(d)

    @staticmethod
    def unregister(domain: str, name: str, version: str) -> bool:
        key = _gen_reg_key(domain, name, version)
        if key not in _reg_defs_by_key:
            return False
        d = _reg_defs_by_key.pop(key)
        del _reg_defs_by_cls[d.class_id]
        _reg_defs_by_pkg[d.root_package].remove(d)
        return True

    @staticmethod
    def get_definition(domain: str, name: str, version: str) -> Definition:
        key = _gen_reg_key(domain, name, version)
        return _reg_defs_by_key.get(key)

    @staticmethod
    def get_definitions(root_pkg: str = None) -> Iterable[Definition]:
        if root_pkg and root_pkg != "*":
            return _reg_defs_by_pkg.get(root_pkg, None)

        return _reg_defs_by_key.values()

    @staticmethod
    def get_definition_by_id(id: str) -> Definition:
        prefix, version = id.split(':')
        tokens = version.split('.')
        if len(tokens) != 3:
            raise NotSupportedError.not_supported_version(
                f"Registry version must be in format of x.y.z, but got {version}"
            )
        major = tokens[0]
        key = f"{prefix}:{major}"
        comp_def = _reg_defs_by_key.get(key)
        if comp_def and comp_def.version == version:
            return comp_def

        return None

    @staticmethod
    def get_definition_by_class(cls: Component | type[Component]) -> Definition:
        if isinstance(cls, Component):
            cls = type(cls)

        cls_id = Definition.to_class_id(cls)
        return _reg_defs_by_cls.get(cls_id)

    @staticmethod
    def get_comp_list_def() -> CompListDef:  # type: ignore
        if len(_comp_list_def.comps) != len(_reg_defs_by_key):
            components = [d.component_def for d in _reg_defs_by_key.values()]
            components = sorted(components, key=lambda k: (k.domain, k.name, k.version))
            _comp_list_def.ClearField('comps')
            _comp_list_def.comps.extend(components)

        return _comp_list_def

    @staticmethod
    def build_comp_list_def(comp_defs: list[Definition] | list[ComponentDef]) -> CompListDef:  # type: ignore
        if comp_defs:
            if isinstance(comp_defs[0], Definition):
                comp_defs = [d.component_def for d in comp_defs]
            comp_defs = sorted(comp_defs, key=lambda k: (k.domain, k.name, k.version))

        comp_list_def = CompListDef(
            name=COMP_LIST_NAME,
            desc=COMP_LIST_DESC,
            version=COMP_LIST_VERSION,
            comps=comp_defs,
        )
        return comp_list_def

    @staticmethod
    def get_comp_list_names() -> Iterable[str]:
        return _reg_defs_by_key.keys()


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
