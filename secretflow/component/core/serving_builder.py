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


from dataclasses import dataclass
from enum import Enum, auto

from secretflow.device.device import PYU, DeviceObject


@dataclass
class SchemaChangeInfo:
    used: set[str]
    derived: set[str]
    deleted: set[str]


@dataclass
class PartyData:
    input_schema: DeviceObject  #  pa.Schema
    output_schema: DeviceObject  # pa.Schema
    kwargs: DeviceObject  # dict
    schema_change: DeviceObject  # SchemaChangeInfo


class ServingOp(Enum):
    ARROW_PROCESSING = "arrow_processing"
    DOT_PRODUCT = "dot_product"
    MERGE_Y = "merge_y"
    TREE_SELECT = "tree_select"
    TREE_MERGE = "tree_merge"
    TREE_ENSEMBLE_PREDICT = "tree_ensemble_predict"


class ServingPhase(Enum):
    PREPROCESSING = auto()
    TRAIN = auto()
    PREDICT = auto()
    POSTPROCESSING = auto()


class ServingNode:
    def __init__(self, name: str, op: ServingOp, phase: ServingPhase) -> None:
        self.exec_id = 0
        self.name = name
        self.op = op
        self.phase = phase
        self.parties: dict[str, PartyData] = {}
        self.parents: list[str] = None

    def add(
        self,
        party: str,
        input_schema: DeviceObject,
        output_schema: DeviceObject,
        kwargs: DeviceObject,
        schema_change: DeviceObject,
    ):
        self.parties[party] = PartyData(
            input_schema=input_schema,
            output_schema=output_schema,
            kwargs=kwargs,
            schema_change=schema_change,
        )

    @staticmethod
    def build_arrow_processing_kwargs(in_schema_ser, out_schema_ser, json_dag):
        return {
            "input_schema_bytes": in_schema_ser,
            "output_schema_bytes": out_schema_ser,
            "trace_content": json_dag,
            "content_json_flag": True,
        }


class DispatchType(Enum):
    DP_ALL = "DP_ALL"
    DP_ANYONE = "DP_ANYONE"
    DP_SPECIFIED = "DP_SPECIFIED"


@dataclass
class ServingExecution:
    dp_type: DispatchType
    session_run: bool
    party_specific_flag: dict[PYU, bool]


class ServingBuilder:
    def __init__(self, pyus: list[PYU]) -> None:
        self.pyus = pyus
        self.nodes: list[ServingNode] = []
        self.executions: list[ServingExecution] = []
        self.new_execution(DispatchType.DP_ALL, False, None)

    def max_id(self) -> int:
        return len(self.nodes)

    def get_last_node_name(self) -> str:
        return self.nodes[-1].name

    def add_node(self, node: ServingNode):
        node.exec_id = len(self.executions) - 1
        self.nodes.append(node)

    def new_execution(
        self,
        dp_type: DispatchType,
        session_run: bool,
        party_specific_flag: dict[PYU, bool] = None,
    ):
        self.executions.append(
            ServingExecution(dp_type, session_run, party_specific_flag)
        )
