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

import pandas as pd
import pytest
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from sklearn.datasets import load_breast_cancer

from secretflow.component.core import (
    DistDataType,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


@pytest.mark.mpc
def test_feature_filter(sf_production_setup_comp):
    alice_input_path = "test_feature_filter/alice.csv"
    bob_input_path = "test_feature_filter/bob.csv"
    output_path = "test_feature_filter/out.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    x = load_breast_cancer()["data"]
    if self_party == "alice":
        ds = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        ds.to_csv(storage.get_writer(alice_input_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(storage.get_writer(bob_input_path), index=False)

    param = build_node_eval_param(
        domain="data_filter",
        name="feature_filter",
        version="1.0.0",
        attrs={
            "input/input_ds/drop_features": ["a1", "b1", "a3", "b13"],
        },
        inputs=[
            DistData(
                name="input",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[output_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
            ),
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)
    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    if self_party == "alice":
        a_out = orc.read_table(storage.get_reader(output_path))
        assert a_out.shape[1] == 13
        assert "a1" not in a_out.columns
        assert "a3" not in a_out.columns

    if self_party == "alice":
        b_out = orc.read_table(storage.get_reader(output_path))
        assert b_out.shape[1] == 13
        assert "b1" not in b_out.columns
        assert "b13" not in b_out.columns
