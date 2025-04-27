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

from secretflow.spec.v1.component_pb2 import CompListDef, ComponentDef

from .component import Component
from .definition import Definition

_reg_defs_by_key: dict[str, Definition] = {}
_reg_defs_by_cls: dict[str, Definition] = {}
_reg_defs_by_pkg: dict[str, list[Definition]] = defaultdict(list)


def _parse_major(version: str) -> str:
    tokens = version.split(".")
    if len(tokens) != 3:
        raise ValueError(f"version must be in format of x.y.z, but got {version}")
    return tokens[0]


def _gen_reg_key(domain: str, name: str, version: str) -> str:
    return f"{domain}/{name}:{_parse_major(version)}"


def _gen_class_id(cls: Component | type[Component]) -> str:
    if isinstance(cls, Component):
        cls = type(cls)
    return f"{cls.__module__}:{cls.__qualname__}"


class Registry:
    @staticmethod
    def register(d: Definition):
        key = _gen_reg_key(d.domain, d.name, d.version)
        if key in _reg_defs_by_key:
            raise ValueError(f"{key} is already registered")
        class_id = _gen_class_id(d.component_cls)
        _reg_defs_by_key[key] = d
        _reg_defs_by_cls[class_id] = d
        _reg_defs_by_pkg[d.root_package].append(d)

    @staticmethod
    def unregister(domain: str, name: str, version: str) -> bool:
        key = _gen_reg_key(domain, name, version)
        if key not in _reg_defs_by_key:
            return False
        d = _reg_defs_by_key.pop(key)
        class_id = _gen_class_id(d.component_cls)
        del _reg_defs_by_cls[class_id]
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
    def get_definition_keys() -> Iterable[str]:
        return _reg_defs_by_key.keys()

    @staticmethod
    def get_definition_by_key(key: str) -> Definition:
        return _reg_defs_by_key.get(key)

    @staticmethod
    def get_definition_by_id(id: str) -> Definition:
        prefix, version = id.split(":")
        key = f"{prefix}:{_parse_major(version)}"
        comp_def = _reg_defs_by_key.get(key)
        if comp_def and comp_def.version == version:
            return comp_def

        return None

    @staticmethod
    def get_definition_by_class(cls: Component | type[Component]) -> Definition:
        class_id = _gen_class_id(cls)
        return _reg_defs_by_cls.get(class_id)


def register(domain: str, version: str, name: str = "", desc: str = None):
    if domain == "" or version == "":
        raise ValueError(
            f"domain<{domain}> and version<{version}> cannot be empty in register"
        )

    def wrap(cls):
        d = Definition(cls, domain, version, name, desc)
        Registry.register(d)
        return cls

    return wrap


COMP_LIST_NAME = "secretflow"
COMP_LIST_DESC = "First-party SecretFlow components."
COMP_LIST_VERSION = "0.0.1"

_comp_list_def = CompListDef(
    name=COMP_LIST_NAME, desc=COMP_LIST_DESC, version=COMP_LIST_VERSION
)


def get_comp_list_def() -> CompListDef:
    definitions = Registry.get_definitions()
    if len(_comp_list_def.comps) != len(definitions):
        res = build_comp_list_def(definitions)
        _comp_list_def.CopyFrom(res)

    return _comp_list_def


def build_comp_list_def(comps: list[Definition] | list[ComponentDef]) -> CompListDef:
    if comps:
        if isinstance(next(iter(comps)), Definition):
            comps = [d.component_def for d in comps]
        comps = sorted(comps, key=lambda k: (k.domain, k.name, k.version))

    comp_list_def = CompListDef(
        name=COMP_LIST_NAME,
        desc=COMP_LIST_DESC,
        version=COMP_LIST_VERSION,
        comps=comps,
    )
    return comp_list_def
