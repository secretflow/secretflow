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

import pyarrow as pa

from secretflow.device.device import PYU


class ServingOp(Enum):
    ARROW_PROCESSING = "arrow_processing"
    DOT_PRODUCT = "dot_product"
    MERGE_Y = "merge_y"
    TREE_SELECT = "tree_select"
    TREE_MERGE = "tree_merge"
    TREE_ENSEMBLE_PREDICT = "tree_ensemble_predict"


class ServingPhase(Enum):
    PREPROCESSING = auto()
    TRAIN_PREDICT = auto()
    POSTPROCESSING = auto()


class ServingNode:
    def __init__(
        self,
        name: str,
        op: ServingOp,
        phase: ServingPhase,
        input_schemas: dict[PYU, pa.Schema] = {},
        output_schemas: dict[PYU, pa.Schema] = {},
        kwargs: dict[PYU, dict] = {},
        parents: list[str] = None,
    ) -> None:
        self.exec_id = 0
        self.name = name
        self.op = op
        self.phase = phase
        self.input_schemas = input_schemas
        self.output_schemas = output_schemas
        self.kwargs = kwargs
        self.parents: list[str] = parents

    def add(
        self,
        pyu: PYU | str,
        input_schema: pa.Schema,
        output_schema: pa.Schema,
        party_kwargs: dict,
    ):
        if isinstance(pyu, str):
            pyu = PYU(pyu)
        self.input_schemas[pyu] = input_schema
        self.output_schemas[pyu] = output_schema
        self.kwargs[pyu] = party_kwargs

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
        session_run: bool = False,
        party_specific_flag: dict[PYU, bool] = None,
    ):
        self.executions.append(
            ServingExecution(dp_type, session_run, party_specific_flag)
        )
