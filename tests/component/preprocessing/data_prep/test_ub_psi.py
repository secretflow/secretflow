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
import tarfile

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
def test_ub_psi(sf_production_setup_comp):
    server_csv_uri = "test_ub_psi/input1.csv"
    client_csv_uri = "test_ub_psi/input2.csv"
    cache_uri = "test_ub_psi/ub_cache_uri"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    input_data_list = [
        {
            "data": {"id1": ["K100", "K101"], "item": ["A1", "A2"]},
            "path": server_csv_uri,
            'party': 'alice',
        },
        {
            "data": {"id2": ["K100", "K102"], "item": ["B1", "B3"]},
            "path": client_csv_uri,
            'party': 'bob',
        },
    ]

    for item in input_data_list:
        if self_party == item['party']:
            pd.DataFrame(item["data"]).to_csv(
                storage.get_writer(item["path"]),
                index=False,
            )

    param = build_node_eval_param(
        domain="data_prep",
        name="unbalance_psi_cache",
        version="1.0.0",
        attrs={'input/input_ds/keys': ["id1"], 'client': ["bob"]},
        inputs=[
            VTable(
                name="input1",
                parties=[
                    VTableParty.from_dict(
                        uri=server_csv_uri,
                        party="alice",
                        format="csv",
                        features={"id1": "str", "item": "str"},
                    )
                ],
            ),
        ],
        output_uris=[cache_uri],
    )

    ub_cache = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(ub_cache.outputs) == 1
    logging.info(f'server cache: {ub_cache.outputs[0]}')

    for data_ref in ub_cache.outputs[0].data_refs:
        if self_party == data_ref.party:
            with storage.get_reader(data_ref.uri) as reader:
                with tarfile.open(fileobj=reader, mode='r:gz') as tar:
                    logging.info(f"\n{self_party} -> output: {tar.getmembers()}\n")

    # unbalance psi
    ub_psi_result_uri = 'ub_psi_result_uri'
    param = build_node_eval_param(
        domain="data_prep",
        name="unbalance_psi",
        version="1.0.0",
        attrs={'input/client_ds/keys': ["id2"], "receiver_parties": ['alice', 'bob']},
        inputs=[
            VTable(
                name="input1",
                parties=[
                    VTableParty.from_dict(
                        uri=client_csv_uri,
                        party="bob",
                        format="csv",
                        features={"id2": "str", "item": "str"},
                    )
                ],
            ),
            ub_cache.outputs[0],
        ],
        output_uris=[ub_psi_result_uri],
    )

    ub_psi_result = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    assert len(ub_psi_result.outputs) == 1
    logging.info(f'client result: {ub_psi_result.outputs[0]}')
    for data_ref in ub_psi_result.outputs[0].data_refs:
        if self_party == data_ref.party:
            table = orc.read_table(storage.get_reader(data_ref.uri))
            logging.info(f"{self_party} table: {table}")


@pytest.mark.mpc
def test_ub_psi_left(sf_production_setup_comp):
    server_csv_uri = "test_ub_psi/input1.csv"
    client_csv_uri = "test_ub_psi/input2.csv"
    cache_uri = "test_ub_psi/ub_cache_uri"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    input_data_list = [
        {
            "data": {"id1": ["K100", "K101"], "item": ["A1", "A2"]},
            "path": server_csv_uri,
            'party': 'alice',
        },
        {
            "data": {"id2": ["K100", "K102"], "item": ["B1", "B3"]},
            "path": client_csv_uri,
            'party': 'bob',
        },
    ]

    for item in input_data_list:
        if self_party == item['party']:
            pd.DataFrame(item["data"]).to_csv(
                storage.get_writer(item["path"]),
                index=False,
            )

    param = build_node_eval_param(
        domain="data_prep",
        name="unbalance_psi_cache",
        version="1.0.0",
        attrs={
            'input/input_ds/keys': ["id1"],
            'client': ["bob"],
        },
        inputs=[
            VTable(
                name="input1",
                parties=[
                    VTableParty.from_dict(
                        uri=server_csv_uri,
                        party="alice",
                        format="csv",
                        features={"id1": "str", "item": "str"},
                    )
                ],
            ),
        ],
        output_uris=[cache_uri],
    )

    ub_cache = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(ub_cache.outputs) == 1
    logging.info(f'server cache: {ub_cache.outputs[0]}')

    for data_ref in ub_cache.outputs[0].data_refs:
        if self_party == data_ref.party:
            with storage.get_reader(data_ref.uri) as reader:
                with tarfile.open(fileobj=reader, mode='r:gz') as tar:
                    logging.info(f"\n{self_party} -> output: {tar.getmembers()}\n")

    # unbalance psi
    ub_psi_result_uri = 'ub_psi_result_uri'
    param = build_node_eval_param(
        domain="data_prep",
        name="unbalance_psi",
        version="1.0.0",
        attrs={
            'input/client_ds/keys': ["id2"],
            "receiver_parties": ['alice', 'bob'],
            "join_type": "left_join",
            "join_type/left_join/left_side": ["alice"],
        },
        inputs=[
            VTable(
                name="input1",
                parties=[
                    VTableParty.from_dict(
                        uri=client_csv_uri,
                        party="bob",
                        format="csv",
                        features={"id2": "str", "item": "str"},
                    )
                ],
            ),
            ub_cache.outputs[0],
        ],
        output_uris=[ub_psi_result_uri],
    )

    ub_psi_result = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    assert len(ub_psi_result.outputs) == 1
    logging.info(f'client result: {ub_psi_result.outputs[0]}')
    for data_ref in ub_psi_result.outputs[0].data_refs:
        if self_party == data_ref.party:
            table = orc.read_table(storage.get_reader(data_ref.uri))
            logging.info(f"{self_party} table: {table}")
