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
import pytest
from pyarrow import orc

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
    write_csv,
)


@pytest.mark.mpc(parties=3)
def test_psi_tp(sf_production_setup_comp):
    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    work_dir = "test_psi_tp"
    output_path = f"{work_dir}/output_ds"
    parties = ["alice", "bob", "carol"]

    input_datas = {
        "alice": {
            "id1": ["K100", "K300", "K200", "K400", "K500"],
            "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
        },
        "bob": {
            "id2": ["K500", "K200", "K300", "K400", "K600", "K700"],
            "feature2": ["DD", "AA", "BB", "CC", "EE", "FF"],
        },
        "carol": {
            "id3": ["K501", "K200", "K300", "K400", "K600", "K700"],
            "feature3": ["DD", "AA", "BB", "CC", "EE", "FF"],
        },
    }

    if self_party in input_datas:
        path = f"{work_dir}/{self_party}"
        with storage.get_writer(path) as w:
            write_csv(input_datas[self_party], w)

    schemas = {
        party: VTableParty.from_dict(
            party=party,
            format="csv",
            uri=f"{work_dir}/{party}",
            features={f"feature{idx+1}": "str", f"id{idx+1}": "str"},
        )
        for idx, party in enumerate(parties)
    }

    param = build_node_eval_param(
        domain="data_prep",
        name="psi_tp",
        version="1.0.0",
        attrs={
            "input/input_ds1/keys1": ["id1"],
            "input/input_ds2/keys2": ["id2"],
            "input/input_ds3/keys3": ["id3"],
        },
        inputs=[VTable(name=party, parties=[s]) for party, s in schemas.items()],
        output_uris=[output_path],
    )

    res = comp_eval(
        param=param, storage_config=storage_config, cluster_config=sf_cluster_config
    )
    assert len(res.outputs) == 1 and len(res.outputs[0].data_refs) == 3
    logging.info(f"res: {res}")

    if self_party in input_datas:
        expected_ids = ["K200", "K300", "K400"]

        idx = parties.index(self_party)
        id_name = f"id{idx+1}"
        dr = res.outputs[0].data_refs[idx]
        assert dr.uri == output_path and dr.format == "orc"
        out_df: pd.DataFrame = orc.read_table(
            storage.get_reader(output_path)
        ).to_pandas()
        assert set(expected_ids) == set(out_df[id_name]), f"{out_df[id_name]}"
