import os

import pandas as pd
from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.data_prep.psi import psi_comp
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from tests.conftest import TEST_STORAGE_ROOT


def test_psi(comp_prod_sf_cluster_config):
    receiver_input_path = "test_psi/receiver_input.csv"
    sender_input_path = "test_psi/sender_input.csv"
    output_path = "test_psi/psi_output.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K300", "K200", "K400", "K500"],
                "item": ["A", "D", "B", "E", "G"],
                "feature1": ["AAA", "DDD", "BBB", "EEE", "GGG"],
            }
        )

        os.makedirs(
            os.path.join(local_fs_wd, "test_psi"),
            exist_ok=True,
        )

        da.to_csv(
            os.path.join(local_fs_wd, receiver_input_path),
            index=False,
        )

    elif self_party == "bob":
        db = pd.DataFrame(
            {
                "id2": ["K500", "K200", "K300", "K400", "K600", "K700"],
                "feature2": ["DD", "AA", "BB", "CC", "EE", "FF"],
            }
        )

        os.makedirs(
            os.path.join(local_fs_wd, "test_psi"),
            exist_ok=True,
        )

        db.to_csv(
            os.path.join(local_fs_wd, sender_input_path),
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
        version="0.0.2",
        attr_paths=[
            "protocol",
            "disable_alignment",
            "ecdh_curve_type",
            "input/receiver_input/key",
            "input/sender_input/key",
        ],
        attrs=[
            Attribute(s="PROTOCOL_ECDH"),
            Attribute(b=False),
            Attribute(s="CURVE_FOURQ"),
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

    pd.testing.assert_frame_equal(
        expected_result_a,
        pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", output_path)),
    )
    pd.testing.assert_frame_equal(
        expected_result_b,
        pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "bob", output_path)),
    )

    output_vt = VerticalTable()

    assert res.outputs[0].meta.Unpack(output_vt)
    assert len(output_vt.schemas) == 2

    assert output_vt.line_count == 4
    assert output_vt.schemas[0].ids == ["id1"]
    assert output_vt.schemas[0].features == ["item", "feature1"]
    assert output_vt.schemas[1].ids == ["id2"]
    assert output_vt.schemas[1].features == ["feature2"]
