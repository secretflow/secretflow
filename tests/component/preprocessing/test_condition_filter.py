import os

import numpy as np
import pandas as pd

from secretflow.component.data_utils import DistDataType, extract_distdata_info
from secretflow.component.preprocessing.condition_filter import condition_filter_comp
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from tests.conftest import TEST_STORAGE_ROOT


def test_condition_filter(comp_prod_sf_cluster_config):
    alice_input_path = "test_condition_filter/alice.csv"
    bob_input_path = "test_condition_filter/bob.csv"
    train_output_path = "test_condition_filter/train.csv"
    test_output_path = "test_condition_filter/test.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

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
            os.path.join(local_fs_wd, "test_condition_filter"),
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
            os.path.join(local_fs_wd, "test_condition_filter"),
            exist_ok=True,
        )

        df_bob.to_csv(
            os.path.join(local_fs_wd, bob_input_path),
            index=False,
        )

    param = NodeEvalParam(
        domain="preprocessing",
        name="condition_filter",
        version="0.0.1",
        attr_paths=[
            'input/in_ds/features',
            'comparator',
            'value_type',
            'bound_value',
            'float_epsilon',
        ],
        attrs=[
            Attribute(ss=['b4']),
            Attribute(s='<'),
            Attribute(s='FLOAT'),
            Attribute(s='11'),
            Attribute(f=0.01),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
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
                id_types=["str"],
                ids=["id2"],
                feature_types=["float32", "str", "float32"],
                features=["b4", "b5", "b6"],
            ),
            TableSchema(
                id_types=["str"],
                ids=["id1"],
                feature_types=["str", "str", "float32"],
                features=["a1", "a2", "a3"],
                label_types=["float32"],
                labels=["y"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    os.makedirs(
        os.path.join(local_fs_wd, "test_condition_filter"),
        exist_ok=True,
    )

    res = condition_filter_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    ds_info = extract_distdata_info(res.outputs[0])
    else_ds_info = extract_distdata_info(res.outputs[1])

    ds_alice = pd.read_csv(
        os.path.join(TEST_STORAGE_ROOT, "alice", ds_info["alice"].uri)
    )
    ds_bob = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "bob", ds_info["bob"].uri))

    ds_else_alice = pd.read_csv(
        os.path.join(TEST_STORAGE_ROOT, "alice", else_ds_info["alice"].uri)
    )
    ds_else_bob = pd.read_csv(
        os.path.join(TEST_STORAGE_ROOT, "bob", else_ds_info["bob"].uri)
    )
    pd.testing.assert_series_equal(
        ds_alice["id1"],
        ds_bob["id2"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        ds_else_alice["id1"],
        ds_else_bob["id2"],
        check_names=False,
    )

    np.testing.assert_equal(ds_alice.shape[0], 2)
    np.testing.assert_equal(ds_else_bob.shape[0], 2)
