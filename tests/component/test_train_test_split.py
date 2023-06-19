import os

import pandas as pd

from secretflow.component.data_utils import DistDataType, extract_distdata_info
from secretflow.component.preprocessing.train_test_split import train_test_split_comp
from secretflow.protos.component.comp_pb2 import Attribute
from secretflow.protos.component.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.protos.component.evaluation_pb2 import NodeEvalParam
from tests.conftest import TEST_STORAGE_ROOT


def test_train_test_split(comp_prod_sf_cluster_config):
    alice_input_path = "test_train_test_split/alice.csv"
    bob_input_path = "test_train_test_split/bob.csv"
    train_output_path = "test_train_test_split/train.csv"
    test_output_path = "test_train_test_split/test.csv"

    self_party = comp_prod_sf_cluster_config.private_config.self_party
    local_fs_wd = comp_prod_sf_cluster_config.private_config.storage_config.local_fs.wd

    if self_party == "alice":
        df_alice = pd.DataFrame(
            {
                "id1": [1, 2, 3, 4],
                "a1": ["K5", "K1", None, "K6"],
                "a2": ["A5", "A1", "A2", "A6"],
                "a3": [5, 1, 2, 6],
                "y": [0, 1, 1, 0],
            }
        )

        os.makedirs(
            os.path.join(local_fs_wd, "test_train_test_split"),
            exist_ok=True,
        )

        df_alice.to_csv(
            os.path.join(local_fs_wd, alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        df_bob = pd.DataFrame(
            {
                "id2": [1, 2, 3, 4],
                "b4": [10.2, 20.5, None, -0.4],
                "b5": ["B3", None, "B9", "B4"],
                "b6": [3, 1, 9, 4],
            }
        )

        os.makedirs(
            os.path.join(local_fs_wd, "test_train_test_split"),
            exist_ok=True,
        )

        df_bob.to_csv(
            os.path.join(local_fs_wd, bob_input_path),
            index=False,
        )

    param = NodeEvalParam(
        domain="preprocessing",
        name="train_test_split",
        version="0.0.1",
        attr_paths=["train_size", "test_size", "random_state", "shuffle"],
        attrs=[
            Attribute(f=0.75),
            Attribute(f=0.25),
            Attribute(i64=1234),
            Attribute(b=False),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                ],
            )
        ],
        output_uris=[
            train_output_path,
            test_output_path,
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                ids=["id1"],
                types=["str", "str", "f32"],
                features=["a1", "a2", "a3"],
                labels=["y"],
            ),
            TableSchema(
                ids=["id2"],
                types=["f32", "str", "f32"],
                features=["b4", "b5", "b6"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    os.makedirs(
        os.path.join(local_fs_wd, "test_train_test_split"),
        exist_ok=True,
    )

    res = train_test_split_comp.eval(param, comp_prod_sf_cluster_config)

    assert len(res.outputs) == 2

    train_info = extract_distdata_info(res.outputs[0])
    test_info = extract_distdata_info(res.outputs[1])

    pd.testing.assert_series_equal(
        pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", train_info["alice"].uri))[
            "id1"
        ],
        pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "bob", train_info["bob"].uri))[
            "id2"
        ],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", test_info["alice"].uri))[
            "id1"
        ],
        pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "bob", test_info["bob"].uri))[
            "id2"
        ],
        check_names=False,
    )
