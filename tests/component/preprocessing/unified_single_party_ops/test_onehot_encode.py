# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import VerticalTable
from secretflow_spec.v1.report_pb2 import Report

import secretflow.compute as sc
from secretflow.component.core import (
    VTable,
    VTableFieldKind,
    VTableParty,
    VTableUtils,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.preprocessing.unified_single_party_ops.onehot_encode import (
    _onehot_encode_fit,
    apply_onehot_rule_on_table,
)


@pytest.mark.mpc
def test_onehot_encode(sf_production_setup_comp):
    alice_input_path = "test_onehot_encode/alice.csv"
    bob_input_path = "test_onehot_encode/bob.csv"
    inplace_encode_path = "test_onehot_encode/inplace_sub.csv"
    rule_path = "test_onehot_encode/onehot.rule"
    report_path = "test_onehot_encode/onehot.report"
    sub_path = "test_onehot_encode/substitution.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        df_alice = pd.DataFrame(
            {
                "id1": [str(i) for i in range(17)],
                "a1": ["K"] + ["F"] * 13 + ["", "M", "N"],
                "a2": [0.1, 0.2, 0.3] * 5 + [0.4] * 2,
                "a3": [1] * 17,
                "y": [0] * 17,
            }
        )
        df_alice.to_csv(
            storage.get_writer(alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        df_bob = pd.DataFrame(
            {
                "id2": [str(i) for i in range(17)],
                "b4": [i for i in range(17)],
                "b5": [i for i in range(17)],
            }
        )
        df_bob.to_csv(
            storage.get_writer(bob_input_path),
            index=False,
        )

    param = build_node_eval_param(
        domain="preprocessing",
        name="onehot_encode",
        version="1.0.0",
        attrs={
            "drop": "first",
            "min_frequency": 0.1,
            "input/input_ds/features": ["a1", "a2", "a3", "b5"],
        },
        inputs=[
            VTable(
                name="input_data",
                parties=[
                    VTableParty.from_dict(
                        uri=bob_input_path,
                        party="bob",
                        format="csv",
                        ids={"id2": "str"},
                        features={"b4": "int32", "b5": "int32"},
                    ),
                    VTableParty.from_dict(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                        ids={"id1": "str"},
                        features={"a1": "str", "a2": "float32", "a3": "int32"},
                        labels={"y": "float32"},
                    ),
                ],
            )
        ],
        output_uris=[
            inplace_encode_path,
            rule_path,
            report_path,
        ],
    )

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 3

    report = Report()
    res.outputs[2].meta.Unpack(report)

    logging.warning(f"....... \n{report}\n.,......")

    meta = VerticalTable()
    res.outputs[0].meta.Unpack(meta)

    logging.warning(f"...meta.... \n{meta}\n.,......")

    param2 = build_node_eval_param(
        domain="preprocessing",
        name="substitution",
        version="1.0.0",
        attrs=None,
        inputs=[param.inputs[0], res.outputs[1]],
        output_uris=[sub_path],
    )

    res = comp_eval(
        param=param2,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    if "alice" == sf_cluster_config.private_config.self_party:
        a_out = orc.read_table(storage.get_reader(sub_path))
        inplace_a_out = orc.read_table(storage.get_reader(inplace_encode_path))

        assert set(a_out.column_names) == set(inplace_a_out.column_names)
        col1 = a_out.column('y').combine_chunks()
        col2 = inplace_a_out.column('y').combine_chunks()
        diff = col1.diff(col2)

        logging.debug(f'diff {diff}')

        for col in a_out.column_names:
            assert a_out.column(col).equals(
                inplace_a_out.column(col)
            ), f"column<{col}> different"

    if "alice" == sf_cluster_config.private_config.self_party:
        b_out = orc.read_table(storage.get_reader(sub_path))
        inplace_b_out = orc.read_table(storage.get_reader(inplace_encode_path))

        assert set(b_out.column_names) == set(inplace_b_out.column_names)
        for col in b_out.column_names:
            assert b_out.column(col).equals(inplace_b_out.column(col))

    # example for how to trace compte without real data
    # for example, we only knows the schema of input
    in_table = sc.Table.from_schema(
        pa.schema(
            [
                VTableUtils.pa_field("a", pa.int32(), VTableFieldKind.FEATURE),
                VTableUtils.pa_field("b", pa.float32(), VTableFieldKind.FEATURE),
                VTableUtils.pa_field("c", pa.string(), VTableFieldKind.FEATURE),
            ]
        )
    )
    # and the onehot rules.
    rules = {"a": [[1], [2, 3]], "c": [["k", "m"]], "b": [[1.11]]}
    # build table from schema and apply rules on it.
    out_table = apply_onehot_rule_on_table(in_table, rules)
    # the data inside out_table is doesn't matter, we care about tracer only
    compute_dag, in_schema, out_schema = out_table.dump_serving_pb("onehot")
    # we have dag and in/out-put's schema, we can build serving arrow op now.
    logging.warning(
        f"compute_dag: \n {compute_dag}\nin_schema:\n{in_schema}\nout_schema:\n{out_schema}"
    )

    r = out_table.dump_runner()
    # trace runner can dump too.
    compute_dag_r, in_schema_r, out_schema_r = r.dump_serving_pb("onehot")
    assert compute_dag_r == compute_dag
    assert in_schema_r == in_schema
    assert out_schema_r == out_schema


def test_onehot_encode_fit():
    df = pd.DataFrame(
        {
            "id1": [str(i) for i in range(17)],
            "a1": ["K"] * 13 + ["F"] + ["", "M", "N"],
            "a2": [0.1, 0.2, 0.3] * 5 + [0.4] * 2,
            "a3": [1] * 17,
            "y": [0] * 17,
        }
    )
    df = pa.Table.from_pandas(df)

    test_datas = [
        {
            "drop": "no_drop",
            "min_frequency": 0,
            "expected": {
                'id1': [[str(i)] for i in range(17)],
                'a1': [['K'], ['F'], [''], ['M'], ['N']],
                'a2': [[0.1], [0.2], [0.3], [0.4]],
                'a3': [[1]],
                'y': [[0]],
            },
        },
        {
            "drop": "no_drop",
            "min_frequency": 0.1,
            "expected": {
                'id1': [[str(i) for i in range(17)]],
                'a1': [['K'], ['F', '', 'M', 'N']],
                'a2': [[0.1], [0.2], [0.3], [0.4]],
                'a3': [[1]],
                'y': [[0]],
            },
        },
        {
            "drop": "mode",
            "min_frequency": 0,
            "expected": {
                'id1': [[str(i + 1)] for i in range(16)],
                'a1': [['F'], [''], ['M'], ['N']],
                'a2': [[0.2], [0.3], [0.4]],
                'a3': [],
                'y': [],
            },
        },
        {
            "drop": "mode",
            "min_frequency": 0.1,
            "expected": {
                'id1': [[str(i + 1) for i in range(16)]],
                'a1': [['F', '', 'M', 'N']],
                'a2': [[0.2], [0.3], [0.4]],
                'a3': [],
                'y': [],
            },
        },
        {
            "drop": "first",
            "min_frequency": 0,
            "expected": {
                'id1': [[str(i + 1)] for i in range(16)],
                'a1': [['F'], [''], ['M'], ['N']],
                'a2': [[0.2], [0.3], [0.4]],
                'a3': [],
                'y': [],
            },
        },
        {
            "drop": "first",
            "min_frequency": 0.1,
            "expected": {
                'id1': [[str(i + 1) for i in range(16)]],
                'a1': [['F', '', 'M', 'N']],
                'a2': [[0.2], [0.3], [0.4]],
                'a3': [],
                'y': [],
            },
        },
    ]

    for item in test_datas:
        drop = item['drop']
        min_frequency = item['min_frequency']
        onehot_rules, drop_rules = _onehot_encode_fit(df, drop, min_frequency)
        assert (
            onehot_rules == item['expected']
        ), f"drop: {drop}, min_frequency: {min_frequency}, rules: {onehot_rules}, drop_rules: {drop_rules}"
