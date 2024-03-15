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


from typing import Dict, List, Set, Tuple

import numpy as np
import pyarrow as pa
from google.protobuf import json_format

from secretflow.component.data_utils import (
    DistDataType,
    extract_table_header,
    model_loads,
)
from secretflow.component.preprocessing.binning.vert_binning import (
    BINNING_RULE_MAX_MAJOR_VERSION,
    BINNING_RULE_MAX_MINOR_VERSION,
)
from secretflow.component.preprocessing.core.version import (
    PREPROCESSING_RULE_MAX_MAJOR_VERSION,
    PREPROCESSING_RULE_MAX_MINOR_VERSION,
)
from secretflow.compute import Table, TraceRunner
from secretflow.device import reveal
from secretflow.preprocessing.binning.vert_bin_substitution import binning_rules_to_sc
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from .graph_builder_manager import GraphBuilderManager


def dump_runner(
    runner: TraceRunner,
    input_schema: Dict[str, np.dtype],
    node_name,
    traced_input: Set[str],
    traced_output: Set[str],
) -> Tuple[bytes, pa.Schema, pa.Schema, bytes, bytes]:
    assert traced_input.issubset(set(input_schema.keys()))
    input_schema = {k: v for k, v in input_schema.items() if k in traced_input}

    dag_pb, dag_input_schema, dag_output_schema = runner.dump_serving_pb(node_name)
    dag_inputs = set(dag_input_schema.names)
    assert dag_inputs.issubset(set(input_schema))

    remain_schema = {k: input_schema[k] for k in input_schema if k not in dag_inputs}
    if len(remain_schema):
        tmp_table = Table.from_schema(remain_schema)
        remain_schema = tmp_table.schema
        for name in remain_schema.names:
            dag_input_schema = dag_input_schema.append(remain_schema.field(name))
            dag_output_schema = dag_output_schema.append(remain_schema.field(name))

    assert traced_input == set(dag_input_schema.names)
    assert traced_output == set(dag_output_schema.names)

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


def dump_runner_schema_info(
    runner: TraceRunner, input_schema: Dict[str, np.dtype]
) -> Tuple[List, List, List]:
    deleted, derived, used = runner.column_changes()
    return used, deleted, derived


def build_empty_dag(
    input_schema: Dict[str, np.dtype],
    node_name,
    traced_input: Set[str],
    traced_output: Set[str],
) -> Tuple[bytes, pa.Schema, pa.Schema, bytes, bytes]:
    assert traced_input.issubset(set(input_schema.keys()))
    input_schema = {k: v for k, v in input_schema.items() if k in traced_input}

    tmp_table = Table.from_schema(input_schema)
    dag_pb, dag_input_schema, dag_output_schema = tmp_table.dump_serving_pb(node_name)

    assert traced_output == set(dag_output_schema.names)

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


def generic_preprocessing_converter(
    ctx,
    node_name: str,
    builder: GraphBuilderManager,
    input_schema: Dict[str, Dict[str, np.dtype]],
    rule_ds: DistData,
    traced_input: Dict[str, Set[str]],
    traced_output: Dict[str, Set[str]],
) -> None:
    runner_objs, _ = model_loads(
        ctx,
        rule_ds,
        PREPROCESSING_RULE_MAX_MAJOR_VERSION,
        PREPROCESSING_RULE_MAX_MINOR_VERSION,
        DistDataType.PREPROCESSING_RULE,
        pyus={p.party: p for p in builder.pyus},
    )
    runner_objs = {r.device: r for r in runner_objs}

    party_kwargs = dict()
    party_input_schemas = dict()
    party_output_schemas = dict()
    for pyu in builder.pyus:
        assert pyu.party in traced_input
        assert pyu.party in traced_output
        if pyu in runner_objs:
            dag, in_schema, out_schema, in_schema_ser, out_schema_ser = pyu(
                dump_runner
            )(
                runner_objs[pyu],
                input_schema[pyu.party],
                node_name,
                traced_input[pyu.party],
                traced_output[pyu.party],
            )
        else:
            dag, in_schema, out_schema, in_schema_ser, out_schema_ser = pyu(
                build_empty_dag
            )(
                input_schema[pyu.party],
                node_name,
                traced_input[pyu.party],
                traced_output[pyu.party],
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


def generic_schema_info(
    ctx,
    builder: GraphBuilderManager,
    input_schema: Dict[str, Dict[str, np.dtype]],
    rule_ds: DistData,
) -> Dict[str, Dict[str, Set[str]]]:
    runner_objs, _ = model_loads(
        ctx,
        rule_ds,
        PREPROCESSING_RULE_MAX_MAJOR_VERSION,
        PREPROCESSING_RULE_MAX_MINOR_VERSION,
        DistDataType.PREPROCESSING_RULE,
        pyus={p.party: p for p in builder.pyus},
    )
    runner_objs = {r.device: r for r in runner_objs}

    party_schema_infos = dict()
    for pyu in builder.pyus:
        if pyu in runner_objs:
            used, deleted, derived = pyu(dump_runner_schema_info)(
                runner_objs[pyu], input_schema[pyu.party]
            )
        else:
            used, deleted, derived = ([], [], [])

        party_schema_infos[pyu.party] = [used, deleted, derived]

    party_schema_infos = reveal(party_schema_infos)
    return {
        p: {"used": set(v[0]), "deleted": set(v[1]), "derived": set(v[2])}
        for p, v in party_schema_infos.items()
    }


def dump_binning_rules(
    rules: Dict,
    input_schema: Dict[str, np.dtype],
    node_name,
    traced_input: Set[str],
    traced_output: Set[str],
) -> Tuple[bytes, pa.Schema, pa.Schema, bytes, bytes]:
    assert traced_input.issubset(set(input_schema.keys()))
    input_schema = {k: v for k, v in input_schema.items() if k in traced_input}

    table = binning_rules_to_sc(rules, input_schema)

    dag_pb, dag_input_schema, dag_output_schema = table.dump_serving_pb(node_name)

    assert traced_output == set(dag_output_schema.names)

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


def dump_binning_schema_info(
    rules: Dict, input_schema: Dict[str, np.dtype]
) -> Tuple[List, List, List]:
    table = binning_rules_to_sc(rules, input_schema)
    deleted, derived, used = table.column_changes()
    return used, deleted, derived


def binning_converter(
    ctx,
    node_name: str,
    builder: GraphBuilderManager,
    input_schema: Dict[str, Dict[str, np.dtype]],
    rule_ds: DistData,
    traced_input: Dict[str, Set[str]],
    traced_output: Dict[str, Set[str]],
) -> None:
    model_objs, _ = model_loads(
        ctx,
        rule_ds,
        BINNING_RULE_MAX_MAJOR_VERSION,
        BINNING_RULE_MAX_MINOR_VERSION,
        DistDataType.BIN_RUNNING_RULE,
        pyus={p.party: p for p in builder.pyus},
    )
    model_objs = {r.device: r for r in model_objs}

    party_kwargs = dict()
    party_input_schemas = dict()
    party_output_schemas = dict()
    for pyu in builder.pyus:
        assert pyu.party in traced_input
        assert pyu.party in traced_output
        if pyu in model_objs:
            dag, in_schema, out_schema, in_schema_ser, out_schema_ser = pyu(
                dump_binning_rules
            )(
                model_objs[pyu],
                input_schema[pyu.party],
                node_name,
                traced_input[pyu.party],
                traced_output[pyu.party],
            )
        else:
            dag, in_schema, out_schema, in_schema_ser, out_schema_ser = pyu(
                build_empty_dag
            )(
                input_schema[pyu.party],
                node_name,
                traced_input[pyu.party],
                traced_output[pyu.party],
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


def binning_schema_info(
    ctx,
    builder: GraphBuilderManager,
    input_schema: Dict[str, Dict[str, np.dtype]],
    rule_ds: DistData,
) -> Dict[str, Dict[str, Set[str]]]:
    model_objs, _ = model_loads(
        ctx,
        rule_ds,
        BINNING_RULE_MAX_MAJOR_VERSION,
        BINNING_RULE_MAX_MINOR_VERSION,
        DistDataType.BIN_RUNNING_RULE,
        pyus={p.party: p for p in builder.pyus},
    )
    model_objs = {r.device: r for r in model_objs}

    party_schema_infos = dict()
    for pyu in builder.pyus:
        if pyu in model_objs:
            used, deleted, derived = pyu(dump_binning_schema_info)(
                model_objs[pyu], input_schema[pyu.party]
            )
        else:
            used, deleted, derived = ([], [], [])

        party_schema_infos[pyu.party] = [used, deleted, derived]

    party_schema_infos = reveal(party_schema_infos)

    return {
        p: {"used": set(v[0]), "deleted": set(v[1]), "derived": set(v[2])}
        for p, v in party_schema_infos.items()
    }


class PreprocessingConverter:
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
        self.node_name = f"preprocessing_{node_id}_{param.name}"
        in_dataset = [d for d in in_ds if d.type == DistDataType.VERTICAL_TABLE]
        assert len(in_dataset) == 1
        self.in_dataset = in_dataset[0]

        self.input_schema = extract_table_header(
            self.in_dataset, load_features=True, load_ids=True, load_labels=True
        )

        assert set(self.input_schema) == set([p.party for p in builder.pyus])

        if param.name == "vert_bin_substitution":
            # binning preprocessing
            rule_ds = [d for d in in_ds if d.type == DistDataType.BIN_RUNNING_RULE]
            assert len(rule_ds) == 1
            self.rule_ds = rule_ds[0]
        else:
            # others preprocessing
            io_ds = in_ds if param.name == "substitution" else out_ds
            rule_ds = [d for d in io_ds if d.type == DistDataType.PREPROCESSING_RULE]
            assert len(rule_ds) == 1
            self.rule_ds = rule_ds[0]

    def schema_info(self) -> Dict[str, Dict[str, Set[str]]]:
        if self.param.name == "vert_bin_substitution":
            # binning preprocessing
            return binning_schema_info(
                self.ctx,
                self.builder,
                self.input_schema,
                self.rule_ds,
            )
        else:
            # others preprocessing
            return generic_schema_info(
                self.ctx,
                self.builder,
                self.input_schema,
                self.rule_ds,
            )

    def convert(
        self, traced_input: Dict[str, Set[str]], traced_output: Dict[str, Set[str]]
    ):
        if self.param.name == "vert_bin_substitution":
            # binning preprocessing
            binning_converter(
                self.ctx,
                self.node_name,
                self.builder,
                self.input_schema,
                self.rule_ds,
                traced_input,
                traced_output,
            )
        else:
            # others preprocessing
            generic_preprocessing_converter(
                self.ctx,
                f"{self.node_name}_{self.rule_ds.name}",
                self.builder,
                self.input_schema,
                self.rule_ds,
                traced_input,
                traced_output,
            )
