import os

import pandas as pd
from sklearn.datasets import load_breast_cancer

from secretflow.component.data_utils import DistDataType, extract_distdata_info
from secretflow.component.feature.vert_woe_binning import (
    vert_woe_binning_comp,
    vert_woe_substitution_comp,
)
from secretflow.protos.component.comp_pb2 import Attribute
from secretflow.protos.component.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.protos.component.evaluation_pb2 import NodeEvalParam
from tests.conftest import TEST_STORAGE_ROOT


def test_woe_binning(comp_prod_sf_cluster_config):
    alice_path = "test_woe_binning/x_alice.csv"
    bob_path = "test_woe_binning/x_bob.csv"
    rule_path = "test_woe_binning/woe_rule"
    output_path = "test_woe_binning/woe.csv"

    self_party = comp_prod_sf_cluster_config.private_config.self_party
    local_fs_wd = comp_prod_sf_cluster_config.private_config.storage_config.local_fs.wd

    ds = load_breast_cancer()
    x, y = ds["data"], ds["target"]
    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, "test_woe_binning"),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(os.path.join(local_fs_wd, alice_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, "test_woe_binning"),
            exist_ok=True,
        )

        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(os.path.join(local_fs_wd, bob_path), index=False)

    bin_param_01 = NodeEvalParam(
        domain="feature",
        name="vert_woe_binning",
        version="0.0.1",
        attr_paths=[
            "secure_device_type",
            "input/input_data/feature_selects",
        ],
        attrs=[
            Attribute(s="heu"),
            Attribute(ss=[f"a{i}" for i in range(12)] + [f"b{i}" for i in range(11)]),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[rule_path],
    )

    bin_param_02 = NodeEvalParam(
        domain="feature",
        name="vert_woe_binning",
        version="0.0.1",
        attr_paths=[
            "secure_device_type",
            "input/input_data/feature_selects",
        ],
        attrs=[
            Attribute(s="spu"),
            Attribute(ss=[f"a{i}" for i in range(11)] + [f"b{i}" for i in range(12)]),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[rule_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["f32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
            TableSchema(
                feature_types=["f32"] * 15,
                features=[f"a{i}" for i in range(15)],
                label_types=["f32"],
                labels=["y"],
            ),
        ],
    )
    bin_param_01.inputs[0].meta.Pack(meta)
    bin_param_02.inputs[0].meta.Pack(meta)

    bin_res = vert_woe_binning_comp.eval(bin_param_01, comp_prod_sf_cluster_config)
    bin_res = vert_woe_binning_comp.eval(bin_param_02, comp_prod_sf_cluster_config)

    sub_param = NodeEvalParam(
        domain="feature",
        name="vert_woe_substitution",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
        inputs=[
            bin_param_01.inputs[0],
            bin_res.outputs[0],
        ],
        output_uris=[output_path],
    )

    sub_res = vert_woe_substitution_comp.eval(sub_param, comp_prod_sf_cluster_config)

    assert len(sub_res.outputs) == 1

    output_info = extract_distdata_info(sub_res.outputs[0])

    alice_out = pd.read_csv(
        os.path.join(TEST_STORAGE_ROOT, "alice", output_info["alice"].uri)
    )
    bob_out = pd.read_csv(
        os.path.join(TEST_STORAGE_ROOT, "bob", output_info["bob"].uri)
    )
    assert alice_out.shape[0] == bob_out.shape[0]

    # import logging

    # logging.warn(f"alice_out \n{alice_out}\n....\n")
    # logging.warn(f"bob_out \n{bob_out}\n....\n")
