import json
import logging
import os

import pandas as pd
import pytest
from google.protobuf.json_format import MessageToJson, Parse
from sklearn.datasets import load_breast_cancer

from secretflow.component.data_utils import DistDataType
from secretflow.component.io.io import io_read_data, io_write_data
from secretflow.component.preprocessing.binning.vert_woe_binning import (
    vert_woe_binning_comp,
)
from secretflow.spec.extend.bin_data_pb2 import Bins
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


@pytest.fixture
def vert_woe_bin_rule(comp_prod_sf_cluster_config):
    alice_path = "test_io/x_alice.csv"
    bob_path = "test_io/x_bob.csv"
    rule_path = "test_io/bin_rule"
    report_path = "test_io/report"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

    ds = load_breast_cancer()
    x, y = ds["data"], ds["target"]
    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, "test_io"),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(os.path.join(local_fs_wd, alice_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, "test_io"),
            exist_ok=True,
        )

        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(os.path.join(local_fs_wd, bob_path), index=False)

    bin_param_01 = NodeEvalParam(
        domain="feature",
        name="vert_woe_binning",
        version="0.0.2",
        attr_paths=[
            "input/input_data/feature_selects",
            "bin_num",
            "input/input_data/label",
        ],
        attrs=[
            Attribute(ss=[f"a{i}" for i in range(2)] + [f"b{i}" for i in range(2)]),
            Attribute(i64=8),
            Attribute(ss=["y"]),
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
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
                label_types=["float32"],
                labels=["y"],
            ),
        ],
    )
    bin_param_01.inputs[0].meta.Pack(meta)

    bin_res = vert_woe_binning_comp.eval(
        param=bin_param_01,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    bin_rule = bin_res.outputs[0]
    return bin_rule


@pytest.fixture
def write_data(vert_woe_bin_rule, comp_prod_sf_cluster_config):
    pb_path = "test_io/rule_pb"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="0.0.1",
        inputs=[vert_woe_bin_rule],
        output_uris=[pb_path],
    )
    read_res = io_read_data.eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    bins_pb = Bins()
    read_res.outputs[0].meta.Unpack(bins_pb)
    write_data = MessageToJson(bins_pb)
    return write_data


def test_no_change_correct(vert_woe_bin_rule, write_data, comp_prod_sf_cluster_config):
    new_rule_path = "test_io/new_bin_rule"
    pb_path = "test_io/rule_pb_unchanged"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    write_param = NodeEvalParam(
        domain="io",
        name="write_data",
        version="0.0.1",
        attr_paths=["write_data", "write_data_type"],
        attrs=[
            Attribute(s=write_data),
            Attribute(s=str(DistDataType.BIN_RUNNING_RULE)),
        ],
        inputs=[vert_woe_bin_rule],
        output_uris=[new_rule_path],
    )
    write_res = io_write_data.eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="0.0.1",
        inputs=[write_res.outputs[0]],
        output_uris=[pb_path],
    )
    read_res = io_read_data.eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    bins_pb_unchanged = Bins()
    read_res.outputs[0].meta.Unpack(bins_pb_unchanged)
    write_data_unchanged = MessageToJson(bins_pb_unchanged)
    #  making an exception for hash
    write_data_dict = json.loads(write_data)
    write_data_unchanged_dict = json.loads(write_data_unchanged)
    write_data_unchanged_dict["modelHash"] = write_data_dict.get("modelHash", "")
    write_data_unchanged = (
        json.dumps(write_data_unchanged_dict).replace("\n", "").replace(" ", "")
    )
    write_data = write_data.replace("\n", "").replace(" ", "")
    logging.info(write_data_unchanged)
    assert (
        write_data_unchanged == write_data
    ), f"No ops, they should be the same {write_data}, {write_data_unchanged}"


def test_merge_one_bin_correct(
    vert_woe_bin_rule, write_data, comp_prod_sf_cluster_config
):
    new_rule_path = "test_io/new_bin_rule"
    pb_path = "test_io/rule_pb_unchanged"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    read_bin_pb = Bins()
    Parse(write_data, read_bin_pb)
    read_bin_pb.variable_bins[0].valid_bins[0].mark_for_merge = True
    read_bin_pb.variable_bins[0].valid_bins[1].mark_for_merge = True
    sample_in_bin_0 = read_bin_pb.variable_bins[0].valid_bins[0].total_count
    sample_in_bin_1 = read_bin_pb.variable_bins[0].valid_bins[1].total_count
    original_bin_num = read_bin_pb.variable_bins[0].valid_bin_count
    write_data = MessageToJson(read_bin_pb)

    write_param = NodeEvalParam(
        domain="io",
        name="write_data",
        version="0.0.1",
        attr_paths=["write_data", "write_data_type"],
        attrs=[
            Attribute(s=write_data),
            Attribute(s=str(DistDataType.BIN_RUNNING_RULE)),
        ],
        inputs=[vert_woe_bin_rule],
        output_uris=[new_rule_path],
    )
    write_res = io_write_data.eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="0.0.1",
        inputs=[write_res.outputs[0]],
        output_uris=[pb_path],
    )
    read_res = io_read_data.eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    bins_pb_changed = Bins()
    read_res.outputs[0].meta.Unpack(bins_pb_changed)
    new_bin_num = bins_pb_changed.variable_bins[0].valid_bin_count
    assert new_bin_num == (
        original_bin_num - 1
    ), f"new_bin num should be {original_bin_num - 1}, but got {new_bin_num}"

    new_num_samples_in_bin = bins_pb_changed.variable_bins[0].valid_bins[0].total_count
    assert new_num_samples_in_bin == (
        sample_in_bin_0 + sample_in_bin_1
    ), f"new sample num in bin should be {sample_in_bin_0 + sample_in_bin_1}, but got {new_num_samples_in_bin}"
