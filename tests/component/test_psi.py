import os

import pandas as pd

from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.psi import psi_comp
from secretflow.protos.component.comp_pb2 import Attribute
from secretflow.protos.component.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow.protos.component.evaluation_pb2 import NodeEvalParam
from tests.conftest import TEST_STORAGE_ROOT


def test_psi(comp_prod_sf_cluster_config):
    receiver_input_path = "test_psi/receiver_input.csv"
    sender_input_path = "test_psi/sender_input.csv"
    output_path = "test_psi/psi_output.csv"
    output_path_2 = "test_psi/psi_output_2.csv"

    self_party = comp_prod_sf_cluster_config.private_config.self_party
    local_fs_wd = comp_prod_sf_cluster_config.private_config.storage_config.local_fs.wd

    if self_party == "alice":
        da = pd.DataFrame(
            {
                "id1": ["K100", "K200", "K300", "K400", "K500"],
                "item": ["A", "B", "D", "E", "G"],
                "feature1": ["AAA", "BBB", "DDD", "EEE", "GGG"],
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
                "id2": ["K200", "K300", "K400", "K500", "K600", "K700"],
                "feature2": ["AA", "BB", "CC", "DD", "EE", "FF"],
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
        domain="preprocessing",
        name="psi",
        version="0.0.1",
        attr_paths=[
            "protocol",
            "sort",
            "broadcast_result",
            "bucket_size",
            "ecdh_curve_type",
            "input/receiver_input/key",
            "input/sender_input/key",
        ],
        attrs=[
            Attribute(s="ECDH_PSI_2PC"),
            Attribute(b=True),
            Attribute(b=True),
            Attribute(i64=1048576),
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
            num_lines=-1,
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
            num_lines=-1,
        ),
    )

    res = psi_comp.eval(param, comp_prod_sf_cluster_config)

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

    assert output_vt.schemas[0].ids == ["id1"]
    assert output_vt.schemas[0].features == ["item", "feature1"]
    assert output_vt.schemas[1].ids == ["id2"]
    assert output_vt.schemas[1].features == ["feature2"]

    # keys are not specified.
    param_2 = NodeEvalParam(
        domain="preprocessing",
        name="psi",
        version="0.0.1",
        attr_paths=[
            "protocol",
            "receiver",
            "sort",
            "broadcast_result",
            "bucket_size",
            "ecdh_curve_type",
        ],
        attrs=[
            Attribute(s="ECDH_PSI_2PC"),
            Attribute(s="alice"),
            Attribute(b=True),
            Attribute(b=True),
            Attribute(i64=1048576),
            Attribute(s="CURVE_FOURQ"),
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
        output_uris=[output_path_2],
    )
    param_2.inputs[0].meta.Pack(
        IndividualTable(
            schema=TableSchema(
                feature_types=["str"] * 3,
                features=["item", "feature1"],
                id_types=["str"],
                ids=["id1"],
            ),
            num_lines=-1,
        ),
    )

    param_2.inputs[1].meta.Pack(
        IndividualTable(
            schema=TableSchema(
                feature_types=["str"] * 2,
                features=["feature2"],
                id_types=["str"],
                ids=["id2"],
            ),
            num_lines=-1,
        ),
    )

    res_2 = psi_comp.eval(param_2, comp_prod_sf_cluster_config)

    assert len(res_2.outputs) == 1

    pd.testing.assert_frame_equal(
        expected_result_a,
        pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", output_path_2)),
    )
    pd.testing.assert_frame_equal(
        expected_result_b,
        pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "bob", output_path_2)),
    )

    output_vt_2 = VerticalTable()

    assert res_2.outputs[0].meta.Unpack(output_vt_2)
    assert len(output_vt_2.schemas) == 2

    assert output_vt_2.schemas[0].ids == ["id1"]
    assert output_vt_2.schemas[0].features == ["item", "feature1"]
    assert output_vt_2.schemas[1].ids == ["id2"]
    assert output_vt_2.schemas[1].features == ["feature2"]
