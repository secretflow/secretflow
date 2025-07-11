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


import pyarrow as pa
import pytest
from pyarrow import orc

from secretflow.component.core import (
    VTable,
    VTableFieldKind,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.core.connector.mock import add_mock_table


@pytest.mark.mpc
def test_data_source(sf_production_setup_comp):
    work_path = f"test_data_source"
    alice_path = f"{work_path}/alice"
    output_path = f"{work_path}/output"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    data = {"ID": ["id1", "id2"], "A": ["s1", "s2"], "B": [1, 2], "C": [0.1, 0.2]}
    tbl = pa.Table.from_pydict(data)
    if self_party == "alice":
        add_mock_table(alice_path, tbl)

    param = build_node_eval_param(
        domain="io",
        name="data_source",
        version="1.0.0",
        attrs={
            "party": ["alice"],
            "uri": "mock:///test_data_source/alice?domaindata_id=test_domaindata_id&datasource_id=test_datasource_id&partition_spec=2024",
            "columns": '{"ID": "ID", "A":"FEATURE", "B": "FEATURE", "C": "LABEL"}',
        },
        inputs=None,
        output_uris=[output_path],
    )
    res = comp_eval(param, storage_config, sf_cluster_config)

    if self_party == "alice":
        out_tbl = VTable.from_distdata(res.outputs[0])
        assert out_tbl.get_party(0).kinds == {
            "ID": VTableFieldKind.ID,
            "A": VTableFieldKind.FEATURE,
            "B": VTableFieldKind.FEATURE,
            "C": VTableFieldKind.LABEL,
        }
        out = orc.read_table(storage.get_reader(output_path))
        assert tbl == out
