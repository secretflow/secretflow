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
from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.data_prep.psi import psi_comp
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_psi(comp_prod_sf_cluster_config):
    receiver_input_path = "test_psi/receiver_input.csv"
    sender_input_path = "test_psi/sender_input.csv"
    output_path = "test_psi/psi_output.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        da.to_csv(
            comp_storage.get_writer(receiver_input_path),
            index=False,
        )

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K500", "K200", "K300", "K400", "K600", "K700"],
                "feature2": ["DD", "AA", "BB", "CC", "EE", "FF"],
            }
        )

        db.to_csv(
            comp_storage.get_writer(sender_input_path),
            index=False,
        )

    expected_result_a = pd.DataFrame(
        {
            "id1": ["K200", "K300", "K400", "K500"],
            "item": ["B", "D", "E", "G"],
            "feature1": ["BBB", "DDD", "EEE", "GGG"],
        }
    )
    expected_result_b = pd.DataFrame(
        {
            "id2": ["K200", "K300", "K400", "K500"],
            "feature2": ["AA", "BB", "CC", "DD"],
        }
    )

    param = NodeEvalParam(
        domain="data_prep",
        name="psi",
        version="0.0.5",
        attr_paths=[
            "protocol",
            "sort_result",
            "ecdh_curve_type",
            "allow_duplicate_keys",
            "input/receiver_input/key",
            "input/sender_input/key",
        ],
        attrs=[
            Attribute(s="PROTOCOL_ECDH"),
            Attribute(b=True),
            Attribute(s="CURVE_FOURQ"),
            Attribute(s="no"),
            Attribute(ss=["id1"]),
            Attribute(ss=["id2"]),
        ],
        inputs=[
            DistData(
                name="receiver_input",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=receiver_input_path, party="alice", format="csv"
                    ),
                ],
            ),
            DistData(
                name="sender_input",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=sender_input_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[output_path],
    )
    param.inputs[0].meta.Pack(
        IndividualTable(
            schema=TableSchema(
                feature_types=["str"] * 3,
                features=["item", "feature1"],
                id_types=["str"],
                ids=["id1"],
            ),
            line_count=-1,
        ),
    )

    param.inputs[1].meta.Pack(
        IndividualTable(
            schema=TableSchema(
                feature_types=["str"] * 2,
                features=["feature2"],
                id_types=["str"],
                ids=["id2"],
            ),
            line_count=-1,
        ),
    )

    res = psi_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    if "alice" == sf_cluster_config.private_config.self_party:
        comp_storage = ComponentStorage(storage_config)
        pd.testing.assert_frame_equal(
            expected_result_a,
            pd.read_csv(comp_storage.get_reader(output_path)),
        )
    if "bob" == sf_cluster_config.private_config.self_party:
        comp_storage = ComponentStorage(storage_config)
        csv_b = pd.read_csv(comp_storage.get_reader(output_path))

        pd.testing.assert_frame_equal(
            expected_result_b,
            csv_b,
        )

    output_vt = VerticalTable()

    assert res.outputs[0].meta.Unpack(output_vt)
    assert len(output_vt.schemas) == 2

    assert output_vt.line_count == 4
    assert output_vt.schemas[0].ids == ["id1"]
    assert output_vt.schemas[0].features == ["item", "feature1"]
    assert output_vt.schemas[1].ids == ["id2"]
    assert output_vt.schemas[1].features == ["feature2"]


def test_psi_left(comp_prod_sf_cluster_config):
    receiver_input_path = "test_psi/receiver_input.csv"
    sender_input_path = "test_psi/sender_input.csv"
    output_path = "test_psi/psi_output.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        da.to_csv(
            comp_storage.get_writer(receiver_input_path),
            index=False,
        )

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K500", "K200", "K300", "K400", "K600", "K700"],
                "feature2": ["DD", "AA", "BB", "CC", "EE", "FF"],
                "feature3": [1 for _ in range(6)],
            }
        )

        db.to_csv(
            comp_storage.get_writer(sender_input_path),
            index=False,
        )

    expected_result_a = pd.DataFrame(
        {
            "id1": ["K200", "K300", "K400", "K500", "K100"],
            "item": ["B", "D", "E", "G", "A"],
            "feature1": ["BBB", "DDD", "EEE", "GGG", "AAA"],
        }
    )
    expected_result_b = pd.DataFrame(
        {
            "id2": ["K200", "K300", "K400", "K500", ""],
            "feature2": ["AA", "BB", "CC", "DD", ""],
            "feature3": [1, 1, 1, 1, 0],
        }
    )

    param = NodeEvalParam(
        domain="data_prep",
        name="psi",
        version="0.0.5",
        attr_paths=[
            "protocol",
            "sort_result",
            "ecdh_curve_type",
            "allow_duplicate_keys",
            "allow_duplicate_keys/yes/join_type",
            "allow_duplicate_keys/yes/join_type/left_join/left_side",
            "input/receiver_input/key",
            "input/sender_input/key",
        ],
        attrs=[
            Attribute(s="PROTOCOL_ECDH"),
            Attribute(b=True),
            Attribute(s="CURVE_FOURQ"),
            Attribute(s="yes"),
            Attribute(s="left_join"),
            Attribute(ss=["alice"]),
            Attribute(ss=["id1"]),
            Attribute(ss=["id2"]),
        ],
        inputs=[
            DistData(
                name="receiver_input",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=receiver_input_path, party="alice", format="csv"
                    ),
                ],
            ),
            DistData(
                name="sender_input",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=sender_input_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[output_path],
    )
    param.inputs[0].meta.Pack(
        IndividualTable(
            schema=TableSchema(
                feature_types=["str"] * 2,
                features=["item", "feature1"],
                id_types=["str"],
                ids=["id1"],
            ),
            line_count=-1,
        ),
    )

    param.inputs[1].meta.Pack(
        IndividualTable(
            schema=TableSchema(
                feature_types=["str", "int"],
                features=["feature2", "feature3"],
                id_types=["str"],
                ids=["id2"],
            ),
            line_count=-1,
        ),
    )

    res = psi_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    if "alice" == sf_cluster_config.private_config.self_party:
        comp_storage = ComponentStorage(storage_config)
        pd.testing.assert_frame_equal(
            expected_result_a,
            pd.read_csv(comp_storage.get_reader(output_path)),
        )
    if "bob" == sf_cluster_config.private_config.self_party:
        comp_storage = ComponentStorage(storage_config)
        csv_b = pd.read_csv(
            comp_storage.get_reader(output_path),
            converters={"id2": str, "feature2": str},
        )

        pd.testing.assert_frame_equal(
            expected_result_b,
            csv_b,
        )

    output_vt = VerticalTable()

    assert res.outputs[0].meta.Unpack(output_vt)
    assert len(output_vt.schemas) == 2

    assert output_vt.line_count == 4
    assert output_vt.schemas[0].ids == ["id1"]
    assert output_vt.schemas[0].features == ["item", "feature1"]
    assert output_vt.schemas[1].ids == ["id2"]
    assert output_vt.schemas[1].features == ["feature2", "feature3"]
