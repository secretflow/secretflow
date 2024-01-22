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


from typing import Dict, List, Tuple

import numpy as np

import secretflow.compute as sc
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
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from .graph_builder_manager import GraphBuilderManager


def dump_runner(
    runner: TraceRunner, input_schema: Dict[str, np.dtype], node_name
) -> Tuple[bytes, bytes, bytes]:
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

    dag_pb = dag_pb.SerializeToString()
    dag_input_schema = dag_input_schema.serialize().to_pybytes()
    dag_output_schema = dag_output_schema.serialize().to_pybytes()

    return (dag_pb, dag_input_schema, dag_output_schema)


def build_empty_dag(
    input_schema: Dict[str, np.dtype], node_name
) -> Tuple[bytes, bytes, bytes]:
    tmp_table = Table.from_schema(input_schema)
    dag_pb, dag_input_schema, dag_output_schema = tmp_table.dump_serving_pb(node_name)

    dag_pb = dag_pb.SerializeToString()
    dag_input_schema = dag_input_schema.serialize().to_pybytes()
    dag_output_schema = dag_output_schema.serialize().to_pybytes()

    return (dag_pb, dag_input_schema, dag_output_schema)


def generic_preprocessing_converter(
    ctx,
    node_name: str,
    builder: GraphBuilderManager,
    input_schema: Dict[str, Dict[str, np.dtype]],
    rule_ds: DistData,
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
    for pyu in builder.pyus:
        if pyu in runner_objs:
            dag, in_schema, out_schema = pyu(dump_runner)(
                runner_objs[pyu], input_schema[pyu.party], node_name
            )
        else:
            dag, in_schema, out_schema = pyu(build_empty_dag)(
                input_schema[pyu.party], node_name
            )

        party_kwargs[pyu] = {
            "input_schema_bytes": in_schema,
            "output_schema_bytes": out_schema,
            "trace_content": dag,
            "content_json_flag": False,
        }

    builder.add_node(node_name, "arrow_processing", party_kwargs)


def binning_rules_to_sc(rules: Dict, input_schema: Dict[str, np.dtype]) -> sc.Table:
    rules = {v['name']: v for v in rules["variables"]}
    assert set(rules).issubset(set(input_schema))

    table = Table.from_schema(input_schema)

    for v in rules:
        col = table.column(v)
        rule = rules[v]
        conds = []
        if rule["type"] == "string":
            conds = [sc.equal(col, c) for c in rule["categories"]]
        else:
            split_points = rule["split_points"]
            if len(split_points) == 0:
                conds = []
            else:
                conds = [sc.less_equal(col, c) for c in split_points]
                conds.append(sc.greater(col, split_points[-1]))

        if conds:
            cases = rule["filling_values"] + [rule["else_filling_value"]]
            cases = list(map(np.float32, cases))
            new_col = sc.case_when(sc.make_struct(*conds), *cases)
            table = table.set_column(table.column_names.index(v), v, new_col)

    return table


def dump_binning_rules(
    rules: Dict, input_schema: Dict[str, np.dtype], node_name
) -> Tuple[bytes, bytes, bytes]:
    table = binning_rules_to_sc(rules, input_schema)

    dag_pb, dag_input_schema, dag_output_schema = table.dump_serving_pb(node_name)

    dag_pb = dag_pb.SerializeToString()
    dag_input_schema = dag_input_schema.serialize().to_pybytes()
    dag_output_schema = dag_output_schema.serialize().to_pybytes()

    return (dag_pb, dag_input_schema, dag_output_schema)


def binning_converter(
    ctx,
    node_name: str,
    builder: GraphBuilderManager,
    input_schema: Dict[str, Dict[str, np.dtype]],
    rule_ds: DistData,
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
    for pyu in builder.pyus:
        if pyu in model_objs:
            dag, in_schema, out_schema = pyu(dump_binning_rules)(
                model_objs[pyu], input_schema[pyu.party], node_name
            )
        else:
            dag, in_schema, out_schema = pyu(build_empty_dag)(
                input_schema[pyu.party], node_name
            )

        party_kwargs[pyu] = {
            "input_schema_bytes": in_schema,
            "output_schema_bytes": out_schema,
            "trace_content": dag,
            "content_json_flag": False,
        }

    builder.add_node(node_name, "arrow_processing", party_kwargs)


def preprocessing_converter(
    ctx,
    builder: GraphBuilderManager,
    node_id: int,
    param: NodeEvalParam,
    in_ds: List[DistData],
    out_ds: List[DistData],
) -> None:
    node_name = f"preprocessing_{node_id}_{param.name}"
    in_dataset = [d for d in in_ds if d.type == DistDataType.VERTICAL_TABLE]
    assert len(in_dataset) == 1
    in_dataset = in_dataset[0]

    input_schema = extract_table_header(
        in_dataset, load_features=True, load_ids=True, load_labels=True
    )

    assert set(input_schema) == set([p.party for p in builder.pyus])

    if param.name == "vert_bin_substitution":
        # binning preprocessing
        rule_ds = [d for d in in_ds if d.type == DistDataType.BIN_RUNNING_RULE]
        assert len(rule_ds) == 1
        rule_ds = rule_ds[0]
        binning_converter(
            ctx,
            node_name,
            builder,
            input_schema,
            rule_ds,
        )
    else:
        # others preprocessing
        if param.name == "substitution":
            rule_ds = [d for d in in_ds if d.type == DistDataType.PREPROCESSING_RULE]
        else:
            rule_ds = [d for d in out_ds if d.type == DistDataType.PREPROCESSING_RULE]
        assert len(rule_ds) == 1, f"rule_ds {rule_ds} ... in_ds {in_ds} ... {out_ds}"
        rule_ds = rule_ds[0]

        generic_preprocessing_converter(
            ctx,
            f"{node_name}_{rule_ds.name}",
            builder,
            input_schema,
            rule_ds,
        )
