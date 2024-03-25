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

from dataclasses import dataclass
from typing import Dict, List

from secretflow.component.data_utils import DistDataType


@dataclass
class TestVersion:
    # ["whl", "public"]
    # public: get sf from docker hub
    # whl: install sf from whl file, use for test unpublished version.
    source: str
    # version info, use for public source
    version: str = None
    # whl path, use for whl source
    whl_paths: List[str] = None


@dataclass
class TestNode:
    party: str
    # for ssh
    hostname: str
    local_fs_path: str
    rayfed_port: int = 62110
    spu_port: int = 62111

    ssh_port: int = 22
    ssh_user: str = "root"
    # if ssh_passwd == None, use host keys
    ssh_passwd: str = None
    nic_name: str = "eth0"

    # docker limits
    # cores
    docker_cpu_limit: int = 0
    # mb
    docker_mem_limit: int = 0


@dataclass
class NetCase:
    name: str
    limit_mb: int = 0
    limit_ms: int = 0


@dataclass
class ClusterCase:
    name: str
    # spu, support only one spu for now
    protocol: str
    field: str
    fxp: int
    spu_parties: List[str]


@dataclass
class DAGInput:
    # [DistDataType.VERTICAL_TABLE, DistDataType.INDIVIDUAL_TABLE]
    # see component.data_utils.DistDataType
    data_type: DistDataType
    data_paths: Dict[str, str]
    feature_columns: Dict[str, List[str]] = None
    feature_types: Dict[str, List[str]] = None
    id_columns: Dict[str, List[str]] = None
    id_types: Dict[str, List[str]] = None
    label_columns: Dict[str, List[str]] = None
    label_types: Dict[str, List[str]] = None


@dataclass
class DataCase:
    name: str
    dag_input_data: Dict[str, DAGInput]

    def get_data(self, name: str) -> DAGInput:
        assert name in self.dag_input_data
        return self.dag_input_data[name]


@dataclass
class TestComp:
    uid: str
    comp_domain: str
    comp_name: str
    comp_version: str
    attrs: Dict[str, object]


class PipelineCase:
    # TODO: move to pb as pipeline def
    def __init__(self, name: str):
        self.name = name
        self.comps: Dict[str, TestComp] = {}
        self.comp_inputs: Dict[str, List[str]] = {}

    def add_comp(self, comp: TestComp, inputs: List[str]) -> None:
        # inputs: [comp1.0, comp2.1, DAGInput.train]
        # comp1.0 means: use comp1's 0-th output as first input
        # DAGInput.train means: use DataCase's "train" DAGInput as third input
        assert comp.uid not in self.comps
        assert comp.uid != "DAGInput"
        self.comps[comp.uid] = comp
        self.comp_inputs[comp.uid] = inputs
