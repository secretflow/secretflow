import logging
import os

import pandas as pd
from sklearn.datasets import load_breast_cancer

from secretflow.component.data_utils import DistDataType, extract_distdata_info
from secretflow.component.io.io import io_read_data
from secretflow.component.preprocessing.binning.vert_binning import (
    vert_bin_substitution_comp,
)
from secretflow.component.preprocessing.binning.vert_woe_binning import (
    vert_woe_binning_comp,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report
from tests.conftest import TEST_STORAGE_ROOT


def test_woe_binning(comp_prod_sf_cluster_config):
    alice_path = "test_woe_binning/x_alice.csv"
    bob_path = "test_woe_binning/x_bob.csv"
    rule_path = "test_woe_binning/bin_rule"
    output_path = "test_woe_binning/woe.csv"
    read_data_path = "test_woe_binning/read_data"
    report_path = "test_vert_binning/report"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

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
        version="0.0.2",
        attr_paths=[
            "secure_device_type",
            "input/input_data/feature_selects",
            "input/input_data/label",
            "report_rules",
        ],
        attrs=[
            Attribute(s="heu"),
            Attribute(ss=[f"a{i}" for i in range(12)] + [f"b{i}" for i in range(11)]),
            Attribute(ss=["y"]),
            Attribute(b=True),
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
        output_uris=[rule_path, report_path],
    )

    bin_param_02 = NodeEvalParam(
        domain="feature",
        name="vert_woe_binning",
        version="0.0.2",
        attr_paths=[
            "secure_device_type",
            "binning_method",
            "input/input_data/feature_selects",
            "input/input_data/label",
            "report_rules",
        ],
        attrs=[
            Attribute(s="spu"),
            Attribute(s="eq_range"),
            Attribute(ss=[f"a{i}" for i in range(11)] + [f"b{i}" for i in range(12)]),
            Attribute(ss=["y"]),
            Attribute(b=True),
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
        output_uris=[rule_path, report_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
            TableSchema(
                feature_types=["float32"] * 16,
                features=[f"a{i}" for i in range(15)] + ["y"],
            ),
        ],
    )
    bin_param_01.inputs[0].meta.Pack(meta)
    bin_param_02.inputs[0].meta.Pack(meta)

    bin_res = vert_woe_binning_comp.eval(
        param=bin_param_01,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    bin_res = vert_woe_binning_comp.eval(
        param=bin_param_02,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(bin_res.outputs) == 2
    comp_ret = Report()
    bin_res.outputs[1].meta.Unpack(comp_ret)
    logging.info("bin_res.outputs[1]: %s", comp_ret)

    sub_param = NodeEvalParam(
        domain="preprocessing",
        name="vert_bin_substitution",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
        inputs=[
            bin_param_01.inputs[0],
            bin_res.outputs[0],
        ],
        output_uris=[output_path],
    )

    sub_res = vert_bin_substitution_comp.eval(
        param=sub_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(sub_res.outputs) == 1

    vt = VerticalTable()
    assert sub_res.outputs[0].meta.Unpack(vt)

    for s in vt.schemas:
        logging.warning(f'schema={s}')
        if 'y' in list(s.labels) or 'y' in list(s.features):
            assert 'y' in list(s.labels) and 'y' not in list(s.features)

    output_info = extract_distdata_info(sub_res.outputs[0])

    alice_out = pd.read_csv(
        os.path.join(TEST_STORAGE_ROOT, "alice", output_info["alice"].uri)
    )
    bob_out = pd.read_csv(
        os.path.join(TEST_STORAGE_ROOT, "bob", output_info["bob"].uri)
    )
    assert alice_out.shape[0] == bob_out.shape[0]

    # logging.warning(f"alice_out \n{alice_out}\n....\n")
    # logging.warning(f"bob_out \n{bob_out}\n....\n")

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
        inputs=[bin_res.outputs[0]],
        output_uris=[read_data_path],
    )

    read_res = io_read_data.eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(read_res.outputs) == 1
