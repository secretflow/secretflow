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

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


@pytest.mark.mpc
def test_train_test_split(sf_production_setup_comp):
    alice_input_path = "test_train_test_split/alice.csv"
    bob_input_path = "test_train_test_split/bob.csv"
    train_output_path = "test_train_test_split/train.csv"
    test_output_path = "test_train_test_split/test.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    input_datasets = {
        "alice": pd.DataFrame(
            {
                "id1": ["1", "2", "3", "4"],
                "a1": ["K5", "K1", None, "K6"],
                "a2": ["A5", "A1", "A2", "A6"],
                "a3": [5, 1, 2, 6],
                "y": [0, 1, 1, 0],
            }
        ),
        "bob": pd.DataFrame(
            {
                "id2": ["1", "2", "3", "4"],
                "b4": [10.2, 20.5, None, -0.4],
                "b5": ["B3", None, "B9", "B4"],
                "b6": [3, 1, 9, 4],
            }
        ),
    }

    if self_party in input_datasets:
        path = f"test_train_test_split/{self_party}.csv"
        df = input_datasets[self_party]
        df.to_csv(storage.get_writer(path), index=False, na_rep="nan")

    param = build_node_eval_param(
        domain="data_prep",
        name="train_test_split",
        version="1.0.0",
        attrs={
            "train_size": 0.75,
            "test_size": 0.25,
            "random_state": 1234,
            "shuffle": False,
        },
        inputs=[
            VTable(
                name="input_data",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                        ids={"id1": "str"},
                        features={"a1": "str", "a2": "str", "a3": "float32"},
                        labels={"y": "float32"},
                    ),
                    VTableParty.from_dict(
                        uri=bob_input_path,
                        party="bob",
                        format="csv",
                        ids={"id2": "str"},
                        features={"b4": "float32", "b5": "str", "b6": "float32"},
                    ),
                ],
            )
        ],
        output_uris=[
            train_output_path,
            test_output_path,
        ],
    )

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    if self_party in input_datasets:
        id_name = "id1" if self_party == "alice" else "id2"
        train_info = VTable.from_distdata(res.outputs[0], columns=[id_name])
        test_info = VTable.from_distdata(res.outputs[1], columns=[id_name])
        train_ids: pd.DataFrame = orc.read_table(
            storage.get_reader(train_info.parties[self_party].uri)
        ).to_pandas()
        test_ids: pd.DataFrame = orc.read_table(
            storage.get_reader(test_info.parties[self_party].uri)
        ).to_pandas()

        input_df = input_datasets[self_party]
        assert list(train_ids[id_name]) == list(input_df[id_name].iloc[0:3])
        assert list(test_ids[id_name]) == list(input_df[id_name].iloc[3])
