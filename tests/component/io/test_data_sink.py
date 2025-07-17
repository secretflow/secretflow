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

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


@pytest.mark.mpc
def test_data_sink_individual_table(sf_production_setup_comp):
    work_path = f"test_data_sink_individual_table"
    alice_path = f"{work_path}/alice.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        data = {"A": ["s1", "s2"], "B": [1, 2], "C": [0.1, 0.2]}
        pd.DataFrame(data).to_csv(storage.get_writer(alice_path), index=False)

    param = build_node_eval_param(
        domain="io",
        name="data_sink",
        version="1.0.0",
        attrs={
            "output_uri": "mock:///test_remote_path?domaindata_id=test_domaindata_id&datasource_id=test_datasource_id&partition_spec=2024",
        },
        inputs=[
            VTable(
                name="data_sink",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_path,
                        party="alice",
                        format="csv",
                        features={"A": "str", "B": "int", "C": "float"},
                    ),
                ],
            )
        ],
    )
    comp_eval(param, storage_config, sf_cluster_config)


@pytest.mark.mpc
def test_data_sink_vertical_table(sf_production_setup_comp):
    work_path = f"test_data_sink_vertical_table"
    alice_path = f"{work_path}/alice.csv"
    bob_path = f"{work_path}/bob.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    if self_party == "alice":
        data = {"A1": ["s1", "s2"], "A2": [1, 2], "A3": [0.1, 0.2]}
        pd.DataFrame(data).to_csv(storage.get_writer(alice_path), index=False)
    elif self_party == "bob":
        data = {"B1": ["s1", "s2"], "B2": [1, 2], "B3": [0.1, 0.2]}
        pd.DataFrame(data).to_csv(storage.get_writer(bob_path), index=False)

    param = build_node_eval_param(
        domain="io",
        name="data_sink",
        version="1.0.0",
        attrs={
            "output_party": ["bob"],
            "output_uri": "mock:///test_remote_path?domaindata_id=test_domaindata_id&datasource_id=test_datasource_id&partition_spec=2024",
        },
        inputs=[
            VTable(
                name="data_sink",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_path,
                        party="alice",
                        format="csv",
                        features={"A1": "str", "A2": "int", "A3": "float"},
                    ),
                    VTableParty.from_dict(
                        uri=bob_path,
                        party="bob",
                        format="csv",
                        features={"B1": "str", "B2": "int", "B3": "float"},
                    ),
                ],
            )
        ],
    )
    comp_eval(param, storage_config, sf_cluster_config)
