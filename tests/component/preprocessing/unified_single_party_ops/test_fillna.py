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

from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.unified_single_party_ops.fillna import (
    SUPPORTED_FILL_NA_METHOD,
    fillna,
)
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


@pytest.mark.parametrize("strategy", SUPPORTED_FILL_NA_METHOD)
def test_fillna(comp_prod_sf_cluster_config, strategy):
    alice_input_path = "test_fillna/alice.csv"
    bob_input_path = "test_fillna/bob.csv"
    rule_path = "test_fillna/fillna.rule"
    sub_path = "test_fillna/substitution.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    if self_party == "alice":
        df_alice = pd.DataFrame(
            {
                "id1": [str(i) for i in range(17)],
                "a1": ["K"] + ["F"] * 14 + [np.nan, "N"],
                "a2": [0.1, np.nan, 0.3] * 5 + [0.4] * 2,
                "a3": [1] * 16 + [0],
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
        name="fillna",
        version="0.0.1",
        attr_paths=[
            'strategy',
            'fill_value_float',
            'input/input_dataset/fill_na_features',
            'missing_value',
            'missing_value_type',
        ],
        attrs=[
            Attribute(s=strategy),
            Attribute(f=99.0),
            Attribute(
                ss=(
                    ["a1", "a2", "b4", "b5"]
                    if strategy == "most_frequent"
                    else ["a2", "b4", "b5"]
                )
            ),
            Attribute(s="axt"),
            Attribute(s="general_na"),
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
            sub_path,
            rule_path,
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

    res = fillna.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    if self_party == "alice":
        a_out = pd.read_csv(comp_storage.get_reader(sub_path))
        logging.warning(f"....... \n{a_out}\n.,......")

        if strategy == "most_frequent":
            assert (
                a_out.isnull().sum().sum() == 0
            ), f"DataFrame contains NaN values, {a_out}"
        else:
            assert (
                a_out.isnull().sum().sum() == 1
            ), f"DataFrame contains more than should be NaN values, {a_out}"

    if self_party == "bob":
        b_out = pd.read_csv(comp_storage.get_reader(sub_path))
        logging.warning(f"....... \n{b_out}\n.,......")

        if strategy == "most_frequent":
            assert (
                b_out.isnull().sum().sum() == 0
            ), f"DataFrame contains NaN values, {b_out}"
        else:
            assert (
                b_out.isnull().sum().sum() == 0
            ), f"DataFrame contains more than should be NaN values, {b_out}"
