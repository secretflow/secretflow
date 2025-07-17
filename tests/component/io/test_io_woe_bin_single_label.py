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

import json

import pandas as pd
import pytest
from google.protobuf.json_format import MessageToJson, Parse
from sklearn.datasets import load_breast_cancer

from secretflow.component.core import (
    DistDataType,
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.spec.extend.bin_data_pb2 import Bins


def gen_vert_woe_bin_rule(sf_production_setup_comp):
    alice_path = "test_io/x_alice_single.csv"
    bob_path = "test_io/x_bob_single.csv"
    out_data_path = "test_io/out_data_single"
    rule_path = "test_io/bin_rule_single"
    report_path = "test_io/report_single"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    ds = load_breast_cancer()
    x, y = ds["data"], ds["target"]
    if self_party == "alice":
        y = pd.DataFrame(y, columns=["y"])
        y.to_csv(storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(storage.get_writer(bob_path), index=False)

    bin_param = build_node_eval_param(
        domain="preprocessing",
        name="vert_woe_binning",
        version="1.0.0",
        attrs={
            "input/input_ds/feature_selects": [f"b{i}" for i in range(2)],
            "bin_num": 8,
            "label": ["y"],
        },
        inputs=[
            VTable(
                name="input_data",
                parties=[
                    VTableParty.from_dict(
                        uri=bob_path,
                        party="bob",
                        format="csv",
                        features={f"b{i}": "float32" for i in range(15)},
                    ),
                    VTableParty.from_dict(
                        uri=alice_path,
                        party="alice",
                        format="csv",
                        features={},
                        labels={"y": "float32"},
                    ),
                ],
            ),
        ],
        output_uris=[out_data_path, rule_path, report_path],
    )

    bin_res = comp_eval(
        param=bin_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    bin_rule = bin_res.outputs[1]
    return bin_rule


def gen_write_data(sf_production_setup_comp, vert_woe_bin_rule):
    pb_path = "test_io/rule_single_pb"
    storage_config, sf_cluster_config = sf_production_setup_comp

    read_param = build_node_eval_param(
        domain="io",
        name="read_data",
        version="1.0.0",
        attrs=None,
        inputs=[vert_woe_bin_rule],
        output_uris=[pb_path],
    )
    read_res = comp_eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    bins_pb = Bins()
    read_res.outputs[0].meta.Unpack(bins_pb)
    write_data = MessageToJson(bins_pb)
    return write_data


@pytest.mark.mpc
def test_no_change_correct(sf_production_setup_comp):
    new_rule_path = "test_io/new_bin_rule_single"
    pb_path = "test_io/rule_single_pb_unchanged"

    vert_woe_bin_rule = gen_vert_woe_bin_rule(sf_production_setup_comp)
    write_data = gen_write_data(sf_production_setup_comp, vert_woe_bin_rule)
    storage_config, sf_cluster_config = sf_production_setup_comp
    write_param = build_node_eval_param(
        domain="io",
        name="write_data",
        version="1.0.0",
        attrs={
            "write_data": write_data,
            "write_data_type": str(DistDataType.BINNING_RULE),
        },
        inputs=[vert_woe_bin_rule],
        output_uris=[new_rule_path],
    )
    write_res = comp_eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    read_param = build_node_eval_param(
        domain="io",
        name="read_data",
        version="1.0.0",
        attrs=None,
        inputs=[write_res.outputs[0]],
        output_uris=[pb_path],
    )
    read_res = comp_eval(
        # In the provided code, `read_param` is a function that is used to build a parameter
        # object for evaluating a specific component or operation related to reading data. This
        # function is used to create the necessary input parameters for reading data from a
        # specific source or location.
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
    assert (
        write_data_unchanged == write_data
    ), f"No ops, they should be the same {write_data}, {write_data_unchanged}"


@pytest.mark.mpc
def test_merge_one_bin_correct(sf_production_setup_comp):
    new_rule_path = "test_io/new_bin_rule_single"
    pb_path = "test_io/rule_single_pb_unchanged"

    vert_woe_bin_rule = gen_vert_woe_bin_rule(sf_production_setup_comp)
    write_data = gen_write_data(sf_production_setup_comp, vert_woe_bin_rule)
    storage_config, sf_cluster_config = sf_production_setup_comp

    read_bin_pb = Bins()
    Parse(write_data, read_bin_pb)
    read_bin_pb.variable_bins[0].valid_bins[0].mark_for_merge = True
    read_bin_pb.variable_bins[0].valid_bins[1].mark_for_merge = True
    sample_in_bin_0 = read_bin_pb.variable_bins[0].valid_bins[0].total_count
    sample_in_bin_1 = read_bin_pb.variable_bins[0].valid_bins[1].total_count
    original_bin_num = read_bin_pb.variable_bins[0].valid_bin_count
    write_data = MessageToJson(read_bin_pb)

    write_param = build_node_eval_param(
        domain="io",
        name="write_data",
        version="1.0.0",
        attrs={
            "write_data": write_data,
            "write_data_type": str(DistDataType.BINNING_RULE),
        },
        inputs=[vert_woe_bin_rule],
        output_uris=[new_rule_path],
    )
    write_res = comp_eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    read_param = build_node_eval_param(
        domain="io",
        name="read_data",
        version="1.0.0",
        attrs=None,
        inputs=[write_res.outputs[0]],
        output_uris=[pb_path],
    )
    read_res = comp_eval(
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
