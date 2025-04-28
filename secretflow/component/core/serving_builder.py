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


import io
from dataclasses import dataclass
from enum import IntEnum, auto

import pyarrow as pa
from google.protobuf import json_format
from secretflow_serving_lib import GraphBuilder, build_serving_tar, compute_trace_pb2

from secretflow.device import HEU, PYU, PYUObject, proxy, wait

from .common.types import BaseEnum
from .storage import Storage


class ServingOp(BaseEnum):
    ARROW_PROCESSING = "arrow_processing"
    DOT_PRODUCT = "dot_product"
    MERGE_Y = "merge_y"
    TREE_SELECT = "tree_select"
    TREE_MERGE = "tree_merge"
    TREE_ENSEMBLE_PREDICT = "tree_ensemble_predict"
    PHE_2P_DOT_PRODUCT = "phe_2p_dot_product"
    PHE_2P_REDUCE = "phe_2p_reduce"
    PHE_2P_DECRYPT_PEER_Y = "phe_2p_decrypt_peer_y"
    PHE_2P_MERGE_Y = "phe_2p_merge_y"


class ServingPhase(IntEnum):
    PREPROCESSING = auto()
    TRAIN_PREDICT = auto()
    POSTPROCESSING = auto()


class ServingNode:
    def __init__(
        self,
        name: str,
        op: ServingOp,
        phase: ServingPhase,
        input_schemas: dict[PYU, pa.Schema] | None = None,
        output_schemas: dict[PYU, pa.Schema] | None = None,
        kwargs: dict[PYU, dict] | None = None,
        parents: list[str] | None = None,
    ) -> None:
        self.exec_id = 0
        self.name = name
        self.op = op
        self.phase = phase
        self.input_schemas = input_schemas if input_schemas else {}
        self.output_schemas = output_schemas if output_schemas else {}
        self.kwargs = kwargs if kwargs else {}
        self.parents = parents

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
    def build_arrow_processing_kwargs(in_schema: pa.Schema, out_schema: pa.Schema, dag):
        if isinstance(in_schema, pa.Schema):
            in_schema_bytes = in_schema.serialize().to_pybytes()
        else:
            in_schema_bytes = in_schema
        if isinstance(out_schema, pa.Schema):
            out_schema_bytes = out_schema.serialize().to_pybytes()
        else:
            out_schema_bytes = out_schema

        if isinstance(dag, compute_trace_pb2.ComputeTrace):
            json_dag = json_format.MessageToJson(dag, indent=0).encode("utf-8")
        else:
            json_dag = dag

        return {
            "input_schema_bytes": in_schema_bytes,
            "output_schema_bytes": out_schema_bytes,
            "trace_content": json_dag,
            "content_json_flag": True,
        }


class DispatchType(BaseEnum):
    DP_ALL = "DP_ALL"
    DP_ANYONE = "DP_ANYONE"
    DP_SPECIFIED = "DP_SPECIFIED"
    DP_SELF = "DP_SELF"
    DP_PEER = "DP_PEER"


@dataclass
class ServingExecution:
    dp_type: DispatchType
    session_run: bool
    party_specific_flag: dict[PYU, bool]


@dataclass
class ServingHEUConfig:
    public_key: bytes
    secret_key: bytes
    scale: int


class ServingBuilder:
    def __init__(self, pyus: list[PYU]) -> None:
        self.pyus = pyus
        self.heu_configs: dict[str, ServingHEUConfig] = None
        self.nodes: list[ServingNode] = []
        self.executions: list[ServingExecution] = []
        self.new_execution(DispatchType.DP_ALL, False, None)

    def max_id(self) -> int:
        return len(self.nodes)

    def get_last_node_name(self) -> str | None:
        if len(self.nodes) > 0:
            return self.nodes[-1].name
        else:
            return None

    def add_node(self, node: ServingNode):
        if len(self.nodes) > 0:
            last = self.nodes[-1]
            assert (
                node.phase >= last.phase
            ), f"invalid phase, {node.name}, {node.phase}, {last.name},{last.phase}"
        node.exec_id = len(self.executions) - 1
        assert node.exec_id >= 0, f"invalid exec_id {node.exec_id}"

        if node.parents is None:
            node.parents = [self.nodes[-1].name] if self.nodes else []
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

    def set_he_config(self, heu_dict: dict[str, HEU]):
        self.heu_configs = {}

        def _key_serialize(key):
            return key.serialize()

        for pyu in self.pyus:
            heu = heu_dict[pyu.party]
            pk = heu.get_participant(pyu.party).public_key.remote()
            sk = heu.get_participant(pyu.party).secret_key.remote()
            pk_bytes = pyu(_key_serialize)(pk)
            sk_bytes = pyu(_key_serialize)(sk)
            self.heu_configs[pyu.party] = ServingHEUConfig(
                pk_bytes, sk_bytes, heu.scale
            )

    def dump_tar_files(self, storage: Storage, name: str, desc: str, uri: str) -> None:
        graph_builders = self._to_graph_builders()
        waits = []

        def dump_io(io_handle: io.BytesIO, uri: str):
            with storage.get_writer(uri) as f:
                f.write(io_handle.getvalue())

        for pyu, builder in graph_builders.items():
            proto = builder.build_proto()
            io_handle = pyu(build_serving_tar)(name, desc, proto)
            waits.append(pyu(dump_io)(io_handle, uri))

        wait(waits)

    def _to_graph_builders(self) -> dict[PYU, GraphBuilder]:
        pyu_builder_actor = proxy(PYUObject)(GraphBuilder)
        graph_builders = {p: pyu_builder_actor(device=p) for p in self.pyus}
        if self.heu_configs:
            waits = []
            for pyu, builder in graph_builders.items():
                heu_conf = self.heu_configs[pyu.party]
                res = builder.set_he_config(
                    heu_conf.public_key, heu_conf.secret_key, heu_conf.scale
                )
                waits.append(res)
            wait(waits)

        curr_exec_id = -1
        for node in self.nodes:
            if node.exec_id != curr_exec_id:
                if node.exec_id > 0:
                    # ignore the first execution, because it has added when create GraphBuilder
                    exec = self.executions[node.exec_id]
                    flags = exec.party_specific_flag
                    waits = []
                    for pyu, builder in graph_builders.items():
                        res = builder.begin_new_execution(
                            dispatch_type=str(exec.dp_type),
                            session_run=exec.session_run,
                            specific_flag=flags[pyu] if flags else False,
                        )
                        waits.append(res)
                    wait(waits)
                curr_exec_id = node.exec_id

            # add node
            waits = []
            for pyu, builder in graph_builders.items():
                kwargs = node.kwargs[pyu]
                res = builder.add_node(
                    name=node.name, parents_name=node.parents, op=str(node.op), **kwargs
                )
                waits.append(res)
            wait(waits)
        return graph_builders
