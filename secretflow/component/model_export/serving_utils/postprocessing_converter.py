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

from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
from google.protobuf import json_format

from secretflow.component.data_utils import DistDataType, extract_table_header
from secretflow.component.eval_param_reader import get_value
from secretflow.component.postprocessing.score_card_transformer import (
    apply_score_card_transformer_on_table,
    score_card_transformer_comp,
)
from secretflow.compute.tracer import Table
from secretflow.spec.v1.component_pb2 import AttrType
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from .graph_builder_manager import GraphBuilderManager


class PostprocessingConverter:
    def __init__(
        self,
        ctx,
        builder: GraphBuilderManager,
        node_id: int,
        param: NodeEvalParam,
        in_ds: List[DistData],
        out_ds: List[DistData],
    ) -> None:
        self.ctx = ctx
        self.builder = builder
        self.param = param
        self.node_name = f"postprocessing_{node_id}_{param.name}"
        in_dataset = [d for d in in_ds if d.type == DistDataType.INDIVIDUAL_TABLE]
        assert len(in_dataset) == 1
        self.in_dataset = in_dataset[0]

        self.input_schema, _ = extract_table_header(
            self.in_dataset, load_features=True, load_ids=True, load_labels=True
        )

    def convert(self):
        if self.param.name == "score_card_transformer":
            score_card_transformer_convert(
                self.ctx,
                self.node_name,
                self.builder,
                self.param,
            )
        else:
            raise AssertionError(
                f"not support postprocessing convert, {self.param.name}"
            )


def parse_score_card_transformer_param(param: NodeEvalParam):
    result = {}
    attrs = {}
    for path, at in zip(list(param.attr_paths), list(param.attrs)):
        attrs[path] = at

    definition = score_card_transformer_comp.definition()
    for at in definition.attrs:
        if at.name in attrs:
            value = attrs[at.name]
        else:
            value = at.atomic.default_value
        result[at.name] = get_value(value, at.type, at.custom_protobuf_cls)

    return result


def dump_score_card_transformer_rule(
    rules: Dict,
    node_name,
) -> Tuple[bytes, pa.Schema, pa.Schema, bytes, bytes]:
    rules["predict_score_name"] = "pred_y"
    rules["predict_name"] = "pred_y"
    input_schema = {"pred_y": np.float64}
    input_table = Table.from_schema(input_schema)
    table = apply_score_card_transformer_on_table(input_table, **rules)

    dag_pb, dag_input_schema, dag_output_schema = table.dump_serving_pb(node_name)

    dag_json = json_format.MessageToJson(dag_pb, indent=0).encode("utf-8")
    dag_input_schema_ser = dag_input_schema.serialize().to_pybytes()
    dag_output_schema_ser = dag_output_schema.serialize().to_pybytes()

    return (
        dag_json,
        dag_input_schema,
        dag_output_schema,
        dag_input_schema_ser,
        dag_output_schema_ser,
    )


def score_card_transformer_convert(
    ctx,
    node_name: str,
    builder: GraphBuilderManager,
    param: NodeEvalParam,
) -> None:
    rules = parse_score_card_transformer_param(param)

    party_kwargs = dict()
    party_input_schemas = dict()
    party_output_schemas = dict()
    for pyu in builder.pyus:
        dag, in_schema, out_schema, in_schema_ser, out_schema_ser = pyu(
            dump_score_card_transformer_rule
        )(
            rules,
            node_name,
        )

        party_input_schemas[pyu] = in_schema
        party_output_schemas[pyu] = out_schema
        party_kwargs[pyu] = {
            "input_schema_bytes": in_schema_ser,
            "output_schema_bytes": out_schema_ser,
            "trace_content": dag,
            "content_json_flag": True,
        }

    builder.add_node(
        node_name,
        "arrow_processing",
        party_input_schemas,
        party_output_schemas,
        party_kwargs,
    )
