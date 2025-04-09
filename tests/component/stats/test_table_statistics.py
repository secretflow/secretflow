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
from secretflow_spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow_spec.v1.report_pb2 import Report

from secretflow.component.core import DistDataType, build_node_eval_param, make_storage
from secretflow.component.entry import comp_eval


def check_report(dd: DistData, target_names=['a', 'b', 'c', 'd']):
    r = Report()
    dd.meta.Unpack(r)
    logging.info(r)
    assert (
        len(r.tabs) == 1
        and len(r.tabs[0].divs) == 1
        and len(r.tabs[0].divs[0].children) == 1
    )
    tbl = r.tabs[0].divs[0].children[0].table
    row_names = [r.name for r in tbl.rows]
    assert row_names == target_names


def test_table_statistics_comp(comp_prod_sf_cluster_config):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    alice_input_path = "test_table_statistics/alice.csv"
    bob_input_path = "test_table_statistics/bob.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    test_data = pd.DataFrame(
        {"a": [9, 6, 5, 5], "b": [5, 5, 6, 7], "c": [1, 1, 2, 4], "d": [11, 55, 1, 99]}
    )
    test_data = test_data.astype("float32")

    if self_party == "alice":
        df_alice = test_data.iloc[:, :2]
        df_alice.to_csv(storage.get_writer(alice_input_path), index=False)
    elif self_party == "bob":
        df_bob = test_data.iloc[:, 2:]
        df_bob.to_csv(storage.get_writer(bob_input_path), index=False)

    param = build_node_eval_param(
        domain="stats",
        name="table_statistics",
        version="1.0.0",
        attrs={"input/input_ds/features": ["a", "b", "c", "d"]},
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                ],
            )
        ],
        output_uris=[""],
    )
    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32", "float32"],
                features=["a", "b"],
            ),
            TableSchema(
                feature_types=["float32", "float32"],
                features=["c", "d"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    check_report(res.outputs[0])


def test_table_statistics_individual_comp(comp_prod_sf_cluster_config):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    alice_input_path = "test_table_statistics/alice.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    test_data = pd.DataFrame(
        {"a": [9, 6, 5, 5], "b": [5, 5, 6, 7], "c": [1, 1, 2, 4], "d": [11, 55, 1, 99]}
    )
    test_data = test_data.astype(dtype="float32")

    if self_party == "alice":
        df_alice = test_data
        df_alice.to_csv(storage.get_writer(alice_input_path), index=False)

    param = build_node_eval_param(
        domain="stats",
        name="table_statistics",
        version="1.0.0",
        attrs={"input/input_ds/features": ["a", "b", "c", "d"]},
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv")
                ],
            )
        ],
        output_uris=[""],
    )
    meta = IndividualTable(
        schema=TableSchema(
            feature_types=["float32", "float32", "float32", "float32"],
            features=["a", "b", "c", "d"],
        )
    )
    param.inputs[0].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    check_report(res.outputs[0])
