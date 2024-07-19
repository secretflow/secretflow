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
import pytest

from secretflow.component.data_utils import DistDataType, extract_distdata_info
from secretflow.component.preprocessing.filter.expr_condition_filter import (
    expr_condition_filter_comp,
    parse_columns,
)
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_expr_condition_filter(comp_prod_sf_cluster_config):
    alice_input_path = "test_condition_filter/alice.csv"
    bob_input_path = "test_condition_filter/bob.csv"
    vertical_output_path = "test_condition_filter/vertical_output.csv"
    vertical_else_path = "test_condition_filter/vertical_else.csv"
    individual_output_path = "test_condition_filter/individual_output.csv"
    individual_else_path = "test_condition_filter/individual_else.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    if self_party == "alice":
        df_alice = pd.DataFrame(
            {
                "id1": [1, 2, 3, 4],
                "a1": ["K5", "K1", None, "K6"],
                "a2": ["A5", "A1", "A2", "A6"],
                "a3": [5, 1, 2, 6],
                "y": [0, 1, 1, 0],
            }
        )
        df_alice.to_csv(
            comp_storage.get_writer(alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        df_bob = pd.DataFrame(
            {
                "id2": [1, 2, 3, 4],
                "b4": [10.2, 20.5, None, -0.4],
                "b5": ["B3", None, "B9", "B4"],
                "b6": [3, 1, 9, 4],
            }
        )
        df_bob.to_csv(
            comp_storage.get_writer(bob_input_path),
            index=False,
        )

    param = NodeEvalParam(
        domain="data_filter",
        name="expr_condition_filter",
        version="0.0.1",
        attr_paths=['expr'],
        attrs=[Attribute(s='b4 < 11')],
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
        output_uris=[vertical_output_path, vertical_else_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=["str"],
                ids=["id2"],
                feature_types=["float32", "str", "float32"],
                features=["b4", "b5", "b6"],
            ),
            TableSchema(
                id_types=["str"],
                ids=["id1"],
                feature_types=["str", "str", "float32"],
                features=["a1", "a2", "a3"],
                label_types=["float32"],
                labels=["y"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    res = expr_condition_filter_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    ds_info = extract_distdata_info(res.outputs[0])
    else_ds_info = extract_distdata_info(res.outputs[1])

    if self_party == "alice":
        ds_alice = pd.read_csv(comp_storage.get_reader(ds_info["alice"].uri))
        ds_else_alice = pd.read_csv(comp_storage.get_reader(else_ds_info["alice"].uri))
        np.testing.assert_equal(ds_alice.shape[0], 2)
        assert list(ds_alice["id1"]) == [1, 4]
        assert list(ds_else_alice["id1"]) == [2, 3]

    if self_party == "bob":
        ds_bob = pd.read_csv(comp_storage.get_reader(ds_info["bob"].uri))
        ds_else_bob = pd.read_csv(comp_storage.get_reader(else_ds_info["bob"].uri))
        np.testing.assert_equal(ds_else_bob.shape[0], 2)
        assert list(ds_bob["id2"]) == [1, 4]
        assert list(ds_else_bob["id2"]) == [2, 3]

    # test errors
    param.ClearField("attrs")
    param.attrs.extend([Attribute(s="b4 < 11 AND a1 IS NOT NULL")])
    with pytest.raises(Exception) as exc_info:
        expr_condition_filter_comp.eval(param, storage_config, sf_cluster_config)
    logging.info(f"Caught expected Exception: {exc_info}")

    # test individual table
    param = NodeEvalParam(
        domain="data_filter",
        name="expr_condition_filter",
        version="0.0.1",
        attr_paths=['expr'],
        attrs=[Attribute(s="b4 < 11 and b5 != 'B4'")],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv")
                ],
            )
        ],
        output_uris=[individual_output_path, individual_else_path],
    )

    meta = IndividualTable(
        schema=TableSchema(
            id_types=["str"],
            ids=["id2"],
            feature_types=["float32", "str", "float32"],
            features=["b4", "b5", "b6"],
        ),
    )
    param.inputs[0].meta.Pack(meta)

    res = expr_condition_filter_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2
    ds_info = extract_distdata_info(res.outputs[0])
    else_ds_info = extract_distdata_info(res.outputs[1])
    if self_party == "bob":
        ds_bob = pd.read_csv(comp_storage.get_reader(ds_info["bob"].uri))
        ds_else_bob = pd.read_csv(comp_storage.get_reader(else_ds_info["bob"].uri))
        assert list(ds_bob["id2"]) == [1]
        assert list(ds_else_bob["id2"]) == [2, 3, 4]


def test_parse_columns():
    columns = parse_columns("(age > 20 AND age < 30) OR (type = 2) OR `select` = 1")
    assert set(columns) == set(['age', 'type', 'select'])
    sql = "(field_A > -3.1415926 and field_A <3.1415926 and field_B =1) or (field_C >= 100 and field_C <= 1000 and field_B != 1) or (field_D ='match_1' and field_E != 'MATCH_2' and field_B =1)"
    columns = parse_columns(sql)
    assert set(columns) == set(['field_A', 'field_B', 'field_C', 'field_D', 'field_E'])
    columns = parse_columns("ABS(level - 10) < 1 AND name LIKE 'XX%'")
    assert set(columns) == set(['level', 'name'])
