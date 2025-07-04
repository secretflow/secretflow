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
from secretflow_spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)

from secretflow.component.core import (
    DistDataType,
    VTable,
    VTableParty,
    assert_almost_equal,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.preprocessing.data_prep.union import Union
from secretflow.utils.errors import InvalidArgumentError


@pytest.mark.mpc
def test_union_individual(sf_production_setup_comp):
    input1_path = "test_union_individual/input1.csv"
    input2_path = "test_union_individual/input2.csv"
    output_path = "test_union_individual/output.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    input_data_list = [
        {
            "data": {"id1": ["K100", "K101"], "item": ["A1", "A2"]},
            "path": input1_path,
        },
        {
            "data": {"id1": ["K200", "K201"], "item": ["B1", "B2"]},
            "path": input2_path,
        },
    ]

    expected_result = pd.DataFrame(
        {
            "id1": ["K100", "K101", "K200", "K201"],
            "item": ["A1", "A2", "B1", "B2"],
        }
    )

    if self_party == 'alice':
        for item in input_data_list:
            pd.DataFrame(item["data"]).to_csv(
                storage.get_writer(item["path"]),
                index=False,
            )

    param = build_node_eval_param(
        domain="data_prep",
        name="union",
        version="1.0.0",
        attrs=None,
        inputs=[
            DistData(
                name="input1",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=input1_path, party="alice", format="csv"),
                ],
            ),
            DistData(
                name="input2",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=input2_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[output_path],
    )

    meta = IndividualTable(
        schema=TableSchema(
            id_types=None,
            ids=None,
            feature_types=["str", "str"],
            features=["id1", "item"],
        )
    )
    param.inputs[0].meta.Pack(meta)
    param.inputs[1].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1
    if self_party == "alice":
        real_result = orc.read_table(storage.get_reader(output_path)).to_pandas()
        logging.warning(f"...individual_res:{self_party}... \n{real_result}\n.....")
        assert_almost_equal(
            expected_result,
            real_result,
            ignore_order=True,
            check_dtype=False,
        )


@pytest.mark.mpc
def test_union_vertical(sf_production_setup_comp):
    alice_input1_path = "test_union_vertical/alice_input1.csv"
    alice_input2_path = "test_union_vertical/alice_input2.csv"
    bob_input1_path = "test_union_vertical/bob_input1.csv"
    bob_input2_path = "test_union_vertical/bob_input2.csv"
    output_path = "test_union_vertical/output.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    input_data_map = {
        "alice": [
            {
                "data": {"id1": ["K100", "K101"], "item1": ["A1", "A2"]},
                "path": alice_input1_path,
            },
            {
                "data": {"id1": ["K200", "K201"], "item1": ["B1", "B2"]},
                "path": alice_input2_path,
            },
        ],
        "bob": [
            {
                "data": {"id2": ["K100", "K101"], "item2": ["A1", "A2"]},
                "path": bob_input1_path,
            },
            {
                "data": {"id2": ["K200", "K201"], "item2": ["B1", "B2"]},
                "path": bob_input2_path,
            },
        ],
    }

    expected_alice = pd.DataFrame(
        {
            "id1": ["K100", "K101", "K200", "K201"],
            "item1": ["A1", "A2", "B1", "B2"],
        }
    )

    expected_bob = pd.DataFrame(
        {
            "id2": ["K100", "K101", "K200", "K201"],
            "item2": ["A1", "A2", "B1", "B2"],
        }
    )

    data_list = input_data_map.get(self_party)
    if data_list is not None:
        for item in data_list:
            pd.DataFrame(item["data"]).to_csv(
                storage.get_writer(item["path"]),
                index=False,
            )

    param = build_node_eval_param(
        domain="data_prep",
        name="union",
        version="1.0.0",
        attrs=None,
        inputs=[
            DistData(
                name="input1",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=alice_input1_path, party="alice", format="csv"
                    ),
                    DistData.DataRef(uri=bob_input1_path, party="bob", format="csv"),
                ],
            ),
            DistData(
                name="input2",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=alice_input2_path, party="alice", format="csv"
                    ),
                    DistData.DataRef(uri=bob_input2_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[output_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=None,
                ids=None,
                feature_types=["str", "str"],
                features=["id1", "item1"],
            ),
            TableSchema(
                id_types=None,
                ids=None,
                feature_types=["str", "str"],
                features=["id2", "item2"],
            ),
        ]
    )
    param.inputs[0].meta.Pack(meta)
    param.inputs[1].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    if self_party in ["alice", "bob"]:
        real_result = orc.read_table(storage.get_reader(output_path)).to_pandas()
        logging.warning(f"\n...vertical_res:{self_party}... \n{real_result}\n.....")

        expected_result = expected_alice if self_party == "alice" else expected_bob

        pd.testing.assert_frame_equal(
            expected_result,
            real_result,
            check_dtype=False,
        )


@pytest.mark.mpc
def test_union_load_table(sf_production_setup_comp):
    input1_path = "test_union_load_table/input1.csv"
    input2_path = "test_union_load_table/input2.csv"
    output_path = "test_union_load_table/output.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    input_data_list = [
        {
            "data": {"id": ["K100", "K101"], "item": ["A1", "A2"], "target": [1, 0]},
            "path": input1_path,
        },
        {
            "data": {"id": ["K200", "K201"], "item": ["B1", "B2"], "target": [0, 1]},
            "path": input2_path,
        },
    ]

    expected_result = pd.DataFrame(
        {
            "item": ["A1", "A2", "B1", "B2"],
            "target": [1, 0, 0, 1],
            "id": ["K100", "K101", "K200", "K201"],
        }
    )

    if self_party == 'alice':
        for item in input_data_list:
            pd.DataFrame(item["data"]).to_csv(
                storage.get_writer(item["path"]),
                index=False,
            )

    param = build_node_eval_param(
        domain="data_prep",
        name="union",
        version="1.0.0",
        attrs=None,
        inputs=[
            DistData(
                name="input1",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=input1_path, party="alice", format="csv"),
                ],
            ),
            DistData(
                name="input2",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=input2_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[output_path],
    )

    meta = IndividualTable(
        schema=TableSchema(
            id_types=["str"],
            ids=["id"],
            feature_types=["str"],
            features=["item"],
            label_types=["int"],
            labels=["target"],
        )
    )
    param.inputs[0].meta.Pack(meta)
    param.inputs[1].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1
    if self_party == "alice":
        real_result = orc.read_table(storage.get_reader(output_path))
        logging.warning(f"...load_table_result:{self_party}... \n{real_result}\n.....")
        assert_almost_equal(
            expected_result,
            real_result,
            ignore_order=True,
            check_dtype=False,
        )


def test_union_error():
    # test input type error
    comp = Union(
        input_ds1=DistData(type=str(DistDataType.VERTICAL_TABLE)),
        input_ds2=DistData(type=str(DistDataType.INDIVIDUAL_TABLE)),
    )
    with pytest.raises(InvalidArgumentError, match="input type mismatch"):
        comp.evaluate(None)

    # test input schema error
    tbl1 = VTable(
        name="table1",
        parties=[
            VTableParty.from_dict(
                party="alice",
                uri="alice_input1.csv",
                format="csv",
                features={"id1": "str"},
            ),
            VTableParty.from_dict(
                party="bob",
                uri="bob_input1.csv",
                format="csv",
                features={"item": "str"},
            ),
        ],
    )
    tbl2 = VTable(
        name="table2",
        parties=[
            VTableParty.from_dict(
                party="alice",
                uri="alice_input1.csv",
                format="csv",
                features={"id1": "str"},
            ),
            VTableParty.from_dict(
                party="bob",
                uri="bob_input1.csv",
                format="csv",
                features={"diff_item_name": "str"},
            ),
        ],
    )
    comp = Union(
        input_ds1=tbl1.to_distdata(),
        input_ds2=tbl2.to_distdata(),
    )
    with pytest.raises(InvalidArgumentError, match="input schema mismatch"):
        comp.evaluate(None)
