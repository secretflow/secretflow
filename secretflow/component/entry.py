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


import platform

from secretflow_spec import Registry
from secretflow_spec.v1.component_pb2 import ComponentDef
from secretflow_spec.v1.data_pb2 import StorageConfig
from secretflow_spec.v1.evaluation_pb2 import NodeEvalParam, NodeEvalResult

from secretflow.spec.extend.cluster_pb2 import SFClusterConfig
from secretflow.version import build_message

from .core.entry import comp_eval as core_comp_eval


def get_comp_def(domain: str, name: str, version: str) -> ComponentDef:
    definition = Registry.get_definition(domain, name, version)
    if definition is not None:
        return definition.component_def
    else:
        raise AttributeError(
            f"{domain}/{name}:{version} is not in component list {Registry.get_definition_keys()}"
        )


def comp_eval(
    param: NodeEvalParam,
    storage_config: StorageConfig,
    cluster_config: SFClusterConfig,
    tracer_report: bool = False,
) -> NodeEvalResult:
    import logging
    import os

    if 'PYTEST_CURRENT_TEST' not in os.environ:
        logging.info(f"\n--\n{build_message()}\n--\n")
        logging.info(f'\n--\n*param* \n\n{param}\n--\n')
        logging.info(f'\n--\n*storage_config* \n\n{storage_config}\n--\n')
        logging.info(f'\n--\n*cluster_config* \n\n{cluster_config}\n--\n')
        logging.info(f'\n--\n*system_info* \n\n{platform.uname()}\n--\n')

    return core_comp_eval(param, storage_config, cluster_config, tracer_report)
