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


from secretflow.component.core.registry import Registry
from secretflow.component.preprocessing.data_prep.psi import psi_comp
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig
from secretflow.spec.v1.component_pb2 import CompListDef, ComponentDef
from secretflow.spec.v1.data_pb2 import StorageConfig
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam, NodeEvalResult
from secretflow.version import build_message

from .core.entry import comp_eval as core_comp_eval

ALL_COMPONENTS = [
    psi_comp,
]

COMP_LIST_NAME = "secretflow"
COMP_LIST_DESC = "First-party SecretFlow components."
COMP_LIST_VERSION = "0.0.1"


def gen_key(domain: str, name: str, version: str) -> str:
    return f"{domain}/{name}:{version}"


def generate_comp_list():
    comp_list = CompListDef()
    comp_list.name = COMP_LIST_NAME
    comp_list.desc = COMP_LIST_DESC
    comp_list.version = COMP_LIST_VERSION
    comp_map = {}
    all_comp_defs = []
    for x in ALL_COMPONENTS:
        x_def = x.definition()
        comp_map[gen_key(x_def.domain, x_def.name, x_def.version)] = x
        all_comp_defs.append(x_def)

    for x in Registry.get_component_defs():
        all_comp_defs.append(x)

    all_comp_defs = sorted(all_comp_defs, key=lambda k: (k.domain, k.name, k.version))
    comp_list.comps.extend(all_comp_defs)
    comp_names = [
        gen_key(x_def.domain, x_def.name, x_def.version) for x_def in comp_list.comps
    ]
    return comp_list, comp_map, comp_names


COMP_LIST, COMP_MAP, COMP_NAMES = generate_comp_list()


def rebuild_comp_list():
    comp_list, comp_map, comp_names = generate_comp_list()
    COMP_LIST.CopyFrom(comp_list)
    COMP_MAP.update(comp_map)
    COMP_NAMES.clear()
    COMP_NAMES.extend(comp_names)


def get_comp_def(domain: str, name: str, version: str) -> ComponentDef:  # type: ignore
    definition = Registry.get_definition(domain, name, version)
    if definition is not None:
        return definition.component_def

    key = gen_key(domain, name, version)
    if key in COMP_MAP:
        return COMP_MAP[key].definition()
    else:
        raise AttributeError(f"key {key} is not in component list {COMP_NAMES}")


def comp_eval(
    param: NodeEvalParam,  # type: ignore
    storage_config: StorageConfig,  # type: ignore
    cluster_config: SFClusterConfig,  # type: ignore
    tracer_report: bool = False,
) -> NodeEvalResult:  # type: ignore
    import logging
    import os

    if 'PYTEST_CURRENT_TEST' not in os.environ:
        logging.info(f"\n--\n{build_message()}\n--\n")
        logging.info(f'\n--\n*param* \n\n{param}\n--\n')
        logging.info(f'\n--\n*storage_config* \n\n{storage_config}\n--\n')
        logging.info(f'\n--\n*cluster_config* \n\n{cluster_config}\n--\n')
    key = gen_key(param.domain, param.name, param.version)
    if key in COMP_MAP:
        comp = COMP_MAP[key]
        res = comp.eval(
            param, storage_config, cluster_config, tracer_report=tracer_report
        )
        logging.info(f'\n--\n*res* \n\n{res}\n--\n')
        return res
    else:
        return core_comp_eval(param, storage_config, cluster_config, tracer_report)
