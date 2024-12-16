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
import logging

import pandas as pd
import pytest
from google.protobuf.json_format import MessageToJson, Parse
from sklearn.datasets import load_breast_cancer

from secretflow.component.core import (
    DistDataType,
    VTable,
    VTableParty,
    build_node_eval_param,
    make_storage,
)
from secretflow.component.entry import comp_eval
from secretflow.spec.extend.bin_data_pb2 import Bins


@pytest.fixture
def vert_bin_rule(comp_prod_sf_cluster_config):
    alice_path = "test_io/x_alice.csv"
    bob_path = "test_io/x_bob.csv"
    out_data_path = "test_io/out_data"
    rule_path = "test_io/bin_rule"
    report_path = "test_io/report"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    ds = load_breast_cancer()
    x, y = ds["data"], ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(storage.get_writer(bob_path), index=False)

    bin_param = build_node_eval_param(
        domain="preprocessing",
        name="vert_binning",
        version="1.0.0",
        attrs={
            "input/input_ds/feature_selects": [f"a{i}" for i in range(2)]
            + [f"b{i}" for i in range(2)],
            "bin_num": 4,
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
                        features={f"a{i}": "float32" for i in range(15)},
                        labels={"y": "float32"},
                    ),
                ],
            )
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


@pytest.fixture
def write_data(vert_bin_rule, comp_prod_sf_cluster_config):
    pb_path = "test_io/rule_pb"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    read_param = build_node_eval_param(
        domain="io",
        name="read_data",
        version="1.0.0",
        attrs=None,
        inputs=[vert_bin_rule],
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


def test_no_change_correct(vert_bin_rule, write_data, comp_prod_sf_cluster_config):
    new_rule_path = "test_io/new_bin_rule"
    pb_path = "test_io/rule_pb_unchanged"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    write_param = build_node_eval_param(
        domain="io",
        name="write_data",
        version="1.0.0",
        attrs={
            "write_data": write_data,
            "write_data_type": str(DistDataType.BINNING_RULE),
        },
        inputs=[vert_bin_rule],
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
    bins_pb_unchanged = Bins()
    read_res.outputs[0].meta.Unpack(bins_pb_unchanged)
    write_data_unchanged = MessageToJson(bins_pb_unchanged)
    # making an exception for hash
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


def test_merge_one_bin_correct(vert_bin_rule, write_data, comp_prod_sf_cluster_config):
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

    write_param = build_node_eval_param(
        domain="io",
        name="write_data",
        version="1.0.0",
        attrs={
            "write_data": write_data,
            "write_data_type": str(DistDataType.BINNING_RULE),
        },
        inputs=[vert_bin_rule],
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

    new_filling_value = bins_pb_changed.variable_bins[0].valid_bins[1].filling_value
    assert (
        new_filling_value == 1
    ), f"filling value is unexpected. expected {1}, got {new_filling_value}"
