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

from secretflow.component.data_utils import DistDataType, extract_distdata_info
from secretflow.component.preprocessing.data_prep.train_test_split import (
    train_test_split_comp,
)
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_train_test_split(comp_prod_sf_cluster_config):
    alice_input_path = "test_train_test_split/alice.csv"
    bob_input_path = "test_train_test_split/bob.csv"
    train_output_path = "test_train_test_split/train.csv"
    test_output_path = "test_train_test_split/test.csv"

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
        domain="data_prep",
        name="train_test_split",
        version="0.0.1",
        attr_paths=["train_size", "test_size", "random_state", "shuffle"],
        attrs=[
            Attribute(f=0.75),
            Attribute(f=0.25),
            Attribute(i64=1234),
            Attribute(b=False),
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
            train_output_path,
            test_output_path,
        ],
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

    res = train_test_split_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    train_info = extract_distdata_info(res.outputs[0])
    test_info = extract_distdata_info(res.outputs[1])

    if self_party == "alice":
        train_ids = pd.read_csv(comp_storage.get_reader(train_info["alice"].uri))["id1"]
        test_ids = pd.read_csv(comp_storage.get_reader(test_info["alice"].uri))["id1"]
        assert list(train_ids) == [1, 2, 3]
        assert list(test_ids) == [4]

    if self_party == "bob":
        train_ids = pd.read_csv(comp_storage.get_reader(train_info["bob"].uri))["id2"]
        test_ids = pd.read_csv(comp_storage.get_reader(test_info["bob"].uri))["id2"]
        assert list(train_ids) == [1, 2, 3]
        assert list(test_ids) == [4]
