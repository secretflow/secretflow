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

import numpy as np
import pandas as pd

import secretflow.compute as sc
from secretflow.component.data_utils import DistDataType, VerticalTableWrapper
from secretflow.component.preprocessing.unified_single_party_ops.onehot_encode import (
    apply_onehot_rule_on_table,
    onehot_encode,
    _onehot_encode_fit,
)
from secretflow.component.preprocessing.unified_single_party_ops.substitution import (
    substitution,
)
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report


def test_onehot_encode(comp_prod_sf_cluster_config):
    alice_input_path = "test_onehot_encode/alice.csv"
    bob_input_path = "test_onehot_encode/bob.csv"
    inplace_encode_path = "test_onehot_encode/inplace_sub.csv"
    rule_path = "test_onehot_encode/onehot.rule"
    report_path = "test_onehot_encode/onehot.report"
    sub_path = "test_onehot_encode/substitution.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

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
            comp_storage.get_writer(alice_input_path),
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
            comp_storage.get_writer(bob_input_path),
            index=False,
        )

    param = NodeEvalParam(
        domain="preprocessing",
        name="onehot_encode",
        version="0.0.3",
        attr_paths=[
            "drop",
            "min_frequency",
            "input/input_dataset/features",
        ],
        attrs=[
            Attribute(s="first"),
            Attribute(f=0.1),
            Attribute(ss=["a1", "a2", "a3", "b5"]),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                ],
            )
        ],
        output_uris=[
            inplace_encode_path,
            rule_path,
            report_path,
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=["str"],
                ids=["id2"],
                feature_types=["int32", "int32"],
                features=["b4", "b5"],
            ),
            TableSchema(
                id_types=["str"],
                ids=["id1"],
                feature_types=["str", "float32", "int32"],
                features=["a1", "a2", "a3"],
                label_types=["float32"],
                labels=["y"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    res = onehot_encode.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 3

    report = Report()
    res.outputs[2].meta.Unpack(report)

    logging.warning(f"....... \n{report}\n.,......")

    meta = VerticalTableWrapper.from_dist_data(res.outputs[0], 0)

    logging.warning(f"...meta.... \n{meta}\n.,......")

    param2 = NodeEvalParam(
        domain="preprocessing",
        name="substitution",
        version="0.0.2",
        inputs=[param.inputs[0], res.outputs[1]],
        output_uris=[sub_path],
    )

    res = substitution.eval(
        param=param2,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    if "alice" == sf_cluster_config.private_config.self_party:
        comp_storage = ComponentStorage(storage_config)
        a_out = pd.read_csv(comp_storage.get_reader(sub_path))
        inplace_a_out = pd.read_csv(comp_storage.get_reader(inplace_encode_path))

        logging.warning(f"....... \n{a_out}\n.,......")

        assert a_out.equals(inplace_a_out)

    if "alice" == sf_cluster_config.private_config.self_party:
        comp_storage = ComponentStorage(storage_config)
        b_out = pd.read_csv(comp_storage.get_reader(sub_path))
        inplace_b_out = pd.read_csv(comp_storage.get_reader(inplace_encode_path))

        assert b_out.equals(inplace_b_out)
        logging.warning(f"....... \n{b_out}\n.,......")

    # example for how to trace compte without real data
    # for example, we only knows the schema of input
    in_table = sc.Table.from_schema({"a": np.int32, "b": np.float32, "c": object})
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


def test_onehot_encode_fit(comp_prod_sf_cluster_config):
    df = pd.DataFrame(
        {
            "id1": [str(i) for i in range(17)],
            "a1": ["K"] * 13 + ["F"] + ["", "M", "N"],
            "a2": [0.1, 0.2, 0.3] * 5 + [0.4] * 2,
            "a3": [1] * 17,
            "y": [0] * 17,
        }
    )

    test_datas = [
        {
            "drop": "no_drop",
            "min_frequency": 0,
            "expected": {
                'id1': [
                    ['0'],
                    ['1'],
                    ['10'],
                    ['11'],
                    ['12'],
                    ['13'],
                    ['14'],
                    ['15'],
                    ['16'],
                    ['2'],
                    ['3'],
                    ['4'],
                    ['5'],
                    ['6'],
                    ['7'],
                    ['8'],
                    ['9'],
                ],
                'a1': [[''], ['F'], ['K'], ['M'], ['N']],
                'a2': [[0.1], [0.2], [0.3], [0.4]],
                'a3': [[1]],
                'y': [[0]],
            },
        },
        {
            "drop": "no_drop",
            "min_frequency": 0.1,
            "expected": {
                'id1': [
                    [
                        '0',
                        '1',
                        '10',
                        '11',
                        '12',
                        '13',
                        '14',
                        '15',
                        '16',
                        '2',
                        '3',
                        '4',
                        '5',
                        '6',
                        '7',
                        '8',
                        '9',
                    ]
                ],
                'a1': [['K'], ['', 'F', 'M', 'N']],
                'a2': [[0.1], [0.2], [0.3], [0.4]],
                'a3': [[1]],
                'y': [[0]],
            },
        },
        {
            "drop": "mode",
            "min_frequency": 0,
            "expected": {
                'id1': [
                    ['1'],
                    ['10'],
                    ['11'],
                    ['12'],
                    ['13'],
                    ['14'],
                    ['15'],
                    ['16'],
                    ['2'],
                    ['3'],
                    ['4'],
                    ['5'],
                    ['6'],
                    ['7'],
                    ['8'],
                    ['9'],
                ],
                'a1': [[''], ['F'], ['M'], ['N']],
                'a2': [[0.2], [0.3], [0.4]],
                'a3': [],
                'y': [],
            },
        },
        {
            "drop": "mode",
            "min_frequency": 0.1,
            "expected": {
                'id1': [
                    [
                        '1',
                        '10',
                        '11',
                        '12',
                        '13',
                        '14',
                        '15',
                        '16',
                        '2',
                        '3',
                        '4',
                        '5',
                        '6',
                        '7',
                        '8',
                        '9',
                    ]
                ],
                'a1': [['', 'F', 'M', 'N']],
                'a2': [[0.2], [0.3], [0.4]],
                'a3': [],
                'y': [],
            },
        },
        {
            "drop": "first",
            "min_frequency": 0,
            "expected": {
                'id1': [
                    ['1'],
                    ['10'],
                    ['11'],
                    ['12'],
                    ['13'],
                    ['14'],
                    ['15'],
                    ['16'],
                    ['2'],
                    ['3'],
                    ['4'],
                    ['5'],
                    ['6'],
                    ['7'],
                    ['8'],
                    ['9'],
                ],
                'a1': [['F'], ['K'], ['M'], ['N']],
                'a2': [[0.2], [0.3], [0.4]],
                'a3': [],
                'y': [],
            },
        },
        {
            "drop": "first",
            "min_frequency": 0.1,
            "expected": {
                'id1': [
                    [
                        '1',
                        '10',
                        '11',
                        '12',
                        '13',
                        '14',
                        '15',
                        '16',
                        '2',
                        '3',
                        '4',
                        '5',
                        '6',
                        '7',
                        '8',
                        '9',
                    ]
                ],
                'a1': [['', 'F', 'M', 'N']],
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
        # logging.warning(
        #     f"drop_type: {drop}, min_frequency: {min_frequency}, rules: {onehot_rules}, drop: {drop_rules}"
        # )
        assert (
            onehot_rules == item['expected']
        ), f"drop: {drop}, min_frequency: {min_frequency}, rules: {onehot_rules}, drop_rules: {drop_rules}"
