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

from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.data_prep.union import union_comp
from secretflow.component.stats.table_statistics import (
    gen_table_statistic_report,
    table_statistics_comp,
)
from secretflow.component.storage.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report
from secretflow.stats.table_statistics import table_statistics


def table_statistics_eval(
    storage_config, sf_cluster_config, inputs, features, test_data
):
    self_party = sf_cluster_config.private_config.self_party
    param = NodeEvalParam(
        domain="stats",
        name="table_statistics",
        version="0.0.2",
        attr_paths=["input/input_data/features"],
        attrs=[
            Attribute(ss=features),
        ],
        inputs=inputs,
        output_uris=[""],
    )

    res = table_statistics_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.warning(f"statistic report of {self_party}:\n {comp_ret}\n......")
    assert comp_ret == gen_table_statistic_report(table_statistics(test_data))


def test_union_individual(comp_prod_sf_cluster_config):
    input1_path = "test_union_individual/input1.csv"
    input2_path = "test_union_individual/input2.csv"
    output_path = "test_union_individual/output.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

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
                comp_storage.get_writer(item["path"]),
                index=False,
            )

    param = NodeEvalParam(
        domain="data_prep",
        name="union",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
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

    res = union_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1
    if self_party == "alice":
        comp_storage = ComponentStorage(storage_config)
        real_result = orc.read_table(comp_storage.get_reader(output_path)).to_pandas()
        logging.warning(f"...individual_res:{self_party}... \n{real_result}\n.....")
        pd.testing.assert_frame_equal(
            expected_result,
            real_result,
            check_dtype=False,
        )

    table_statistics_eval(
        storage_config=storage_config,
        sf_cluster_config=sf_cluster_config,
        inputs=res.outputs,
        features=["id1", "item"],
        test_data=pd.DataFrame(
            {"id1": ["K100", "K101", "K200", "K201"], "item": ["A1", "A2", "B1", "B2"]},
        ),
    )


def test_union_vertical(comp_prod_sf_cluster_config):
    alice_input1_path = "test_union_vertical/alice_input1.csv"
    alice_input2_path = "test_union_vertical/alice_input2.csv"
    bob_input1_path = "test_union_vertical/bob_input1.csv"
    bob_input2_path = "test_union_vertical/bob_input2.csv"
    output_path = "test_union_vertical/output.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

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
                comp_storage.get_writer(item["path"]),
                index=False,
            )

    param = NodeEvalParam(
        domain="data_prep",
        name="union",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
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

    res = union_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    if self_party in ["alice", "bob"]:
        comp_storage = ComponentStorage(storage_config)
        real_result = orc.read_table(comp_storage.get_reader(output_path)).to_pandas()
        logging.warning(f"\n...vertical_res:{self_party}... \n{real_result}\n.....")

        expected_result = expected_alice if self_party == "alice" else expected_bob

        pd.testing.assert_frame_equal(
            expected_result,
            real_result,
            check_dtype=False,
        )

    table_statistics_eval(
        storage_config=storage_config,
        sf_cluster_config=sf_cluster_config,
        inputs=res.outputs,
        features=["id1", "item1", "id2", "item2"],
        test_data=pd.DataFrame(
            {
                "id1": ["K100", "K101", "K200", "K201"],
                "item1": ["A1", "A2", "B1", "B2"],
                "id2": ["K100", "K101", "K200", "K201"],
                "item2": ["A1", "A2", "B1", "B2"],
            },
        ),
    )


def test_union_load_table(comp_prod_sf_cluster_config):
    input1_path = "test_union_load_table/input1.csv"
    input2_path = "test_union_load_table/input2.csv"
    output_path = "test_union_load_table/output.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

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
                comp_storage.get_writer(item["path"]),
                index=False,
            )

    param = NodeEvalParam(
        domain="data_prep",
        name="union",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
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

    res = union_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1
    if self_party == "alice":
        comp_storage = ComponentStorage(storage_config)
        real_result = orc.read_table(comp_storage.get_reader(output_path)).to_pandas()
        logging.warning(f"...load_table_result:{self_party}... \n{real_result}\n.....")
        pd.testing.assert_frame_equal(
            expected_result,
            real_result,
            check_dtype=False,
        )


def test_union_error(comp_prod_sf_cluster_config):
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    alice_input1_path = "test_union_error/alice_input1.csv"
    alice_input2_path = "test_union_error/alice_input2.csv"
    bob_input1_path = "test_union_error/bob_input1.csv"
    bob_input2_path = "test_union_error/bob_input2.csv"
    output_path = "test_union_error/output.csv"

    if self_party == "alice":
        # different schema
        pd.DataFrame({"id1": ["K100"]}).to_csv(
            comp_storage.get_writer(alice_input1_path),
            index=False,
        )
        pd.DataFrame({"diff_id": ["K200"]}).to_csv(
            comp_storage.get_writer(alice_input2_path),
            index=False,
        )
    elif self_party == "bob":
        pd.DataFrame({"item": ["item1"]}).to_csv(
            comp_storage.get_writer(bob_input1_path),
            index=False,
        )
        pd.DataFrame({"item": ["item2"]}).to_csv(
            comp_storage.get_writer(bob_input2_path),
            index=False,
        )

    param = NodeEvalParam(
        domain="data_prep",
        name="union",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
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
                feature_types=["str"],
                features=["id1"],
            ),
            TableSchema(
                id_types=None,
                ids=None,
                feature_types=["str"],
                features=["item"],
            ),
        ]
    )
    param.inputs[0].meta.Pack(meta)
    # reset schema
    meta.schemas[0].features[:] = []
    meta.schemas[0].features.append("diff_id")
    param.inputs[1].meta.Pack(meta)

    with pytest.raises(AssertionError, match="table meta info missmatch") as exc_info:
        union_comp.eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )
    logging.warning(f"Caught expected AssertionError: {exc_info}")
