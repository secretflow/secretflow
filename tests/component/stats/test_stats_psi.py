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

import logging

import numpy as np
import pandas as pd
import pytest
from secretflow_spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow_spec.v1.report_pb2 import Report
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from secretflow.component.core import (
    DistDataType,
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.stats.stats_psi import (
    calculate_stats_psi_one_feature,
    get_bin_counts_one_feature,
)

# good psi [0, 0.1], notably significant change: [0, 0.25], substantial variation: [0.25, ]
good_psi_threshold = 0.1


def gen_vert_bin_rule(sf_production_setup_comp):
    alice_path = "test_vert_binning/x_alice.csv"
    bob_path = "test_vert_binning/x_bob.csv"
    alice_test_path = "test_vert_binning/x_alice_test.csv"
    bob_test_path = "test_vert_binning/x_bob_test.csv"
    out_data_path = "test_vert_binning/out_data"
    rule_path = "test_vert_binning/bin_rule"
    report_path = "test_vert_binning/report"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    # shape (569, 30)
    ds = load_breast_cancer()
    x, y = ds["data"], ds["target"]
    # total size 569, train size 398, test size 171,
    X_train, X_test, _, _ = train_test_split(x, y, test_size=0.3, random_state=42)

    if self_party == "alice":
        ds = pd.DataFrame(X_train[:, :2], columns=[f"a{i}" for i in range(2)])
        new_row = pd.DataFrame(np.nan, columns=ds.columns, index=[len(ds)])
        ds = pd.concat([ds, new_row], ignore_index=True)
        ds.to_csv(storage.get_writer(alice_path), index=False, na_rep="nan")
        ds_test = pd.DataFrame(X_test[:, :2], columns=[f"a{i}" for i in range(2)])
        ds_test = pd.concat([ds_test, new_row], ignore_index=True)
        ds_test.to_csv(storage.get_writer(alice_test_path), index=False, na_rep="nan")

    elif self_party == "bob":
        ds = pd.DataFrame(X_train[:, 2:3], columns=[f"b{i}" for i in range(1)])
        new_row = pd.DataFrame(np.nan, columns=ds.columns, index=[len(ds)])
        ds = pd.concat([ds, new_row], ignore_index=True)
        ds.to_csv(storage.get_writer(bob_path), index=False, na_rep="nan")
        ds_test = pd.DataFrame(X_test[:, 2:3], columns=[f"b{i}" for i in range(1)])
        ds_test = pd.concat([ds_test, new_row], ignore_index=True)
        ds_test.to_csv(storage.get_writer(bob_test_path), index=False, na_rep="nan")

    bin_param = build_node_eval_param(
        domain="preprocessing",
        name="vert_binning",
        version="1.0.0",
        attrs={
            "input/input_ds/feature_selects": ['a0', 'a1', 'b0'],
            "report_rules": True,
        },
        inputs=[
            VTable(
                name="input_data",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_path,
                        party="alice",
                        format="csv",
                        features={f"a{i}": "float32" for i in range(2)},
                    ),
                    VTableParty.from_dict(
                        uri=bob_path,
                        party="bob",
                        format="csv",
                        features={f"b{i}": "float32" for i in range(1)},
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

    assert len(bin_res.outputs) == 3
    comp_ret = Report()
    bin_res.outputs[1].meta.Unpack(comp_ret)
    bin_rule = bin_res.outputs[1]
    return bin_rule


@pytest.mark.mpc
def test_stats_psi_input_bin_rule(sf_production_setup_comp):
    # the params in the following must be the same in vert_bin_rule
    alice_path = "test_vert_binning/x_alice.csv"
    bob_path = "test_vert_binning/x_bob.csv"
    alice_test_path = "test_vert_binning/x_alice_test.csv"
    bob_test_path = "test_vert_binning/x_bob_test.csv"
    report_path = "test_vert_binning/report"
    vert_bin_rule = gen_vert_bin_rule(sf_production_setup_comp)
    storage_config, sf_cluster_config = sf_production_setup_comp
    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 1,
                features=[f"b{i}" for i in range(1)],
            ),
            TableSchema(
                feature_types=["float32"] * 2,
                features=[f"a{i}" for i in range(2)],
            ),
        ],
    )

    param = build_node_eval_param(
        domain="stats",
        name="stats_psi",
        version="1.0.0",
        attrs={
            'input/input_base_ds/feature_selects': [f"a{i}" for i in range(2)]
            + [f"b{i}" for i in range(1)],
        },
        inputs=[
            DistData(
                name="input_base_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                ],
            ),
            DistData(
                name="input_test_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_test_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_test_path, party="alice", format="csv"),
                ],
            ),
            vert_bin_rule,
        ],
        output_uris=[report_path],
    )

    param.inputs[0].meta.Pack(meta)
    param.inputs[1].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
        tracer_report=True,
    )

    logging.info(f"train tracer_report {res['tracer_report']}")
    comp_ret = Report()
    res = res["eval_result"]
    res.outputs[0].meta.Unpack(comp_ret)
    logging.info(comp_ret)


def gen_vert_woe_bin_rule(sf_production_setup_comp):
    alice_path = "test_io/x_alice.csv"
    bob_path = "test_io/x_bob.csv"
    out_path = "test_io/out.csv"
    rule_path = "test_io/bin_rule"
    report_path = "test_io/report"

    storage_config, sf_cluster_config = sf_production_setup_comp
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
        name="vert_woe_binning",
        version="1.0.0",
        attrs={
            "input/input_ds/feature_selects": ['a0', 'a1', 'b0'],
            "label": ["y"],
            "bin_num": 8,
        },
        inputs=[
            VTable(
                name="input_data",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_path,
                        party="alice",
                        format="csv",
                        features={f"a{i}": "float32" for i in range(15)},
                        labels={"y": "float32"},
                    ),
                    VTableParty.from_dict(
                        uri=bob_path,
                        party="bob",
                        format="csv",
                        features={f"b{i}": "float32" for i in range(15)},
                    ),
                ],
            ),
        ],
        output_uris=[out_path, rule_path, report_path],
    )

    bin_res = comp_eval(
        param=bin_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    bin_rule = bin_res.outputs[1]
    return bin_rule


@pytest.mark.mpc
def test_stats_psi_input_woe_bin_rule(sf_production_setup_comp):
    # the params in the following must be the same in vert_bin_rule
    alice_path = "test_io/x_alice.csv"
    bob_path = "test_io/x_bob.csv"
    report_path = "test_io/report"
    vert_woe_bin_rule = gen_vert_woe_bin_rule(sf_production_setup_comp)
    storage_config, sf_cluster_config = sf_production_setup_comp
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

    param = build_node_eval_param(
        domain="stats",
        name="stats_psi",
        version="1.0.0",
        attrs={
            'input/input_base_ds/feature_selects': [f"a{i}" for i in range(2)]
            + [f"b{i}" for i in range(1)],
        },
        inputs=[
            DistData(
                name="input_base_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                ],
            ),
            DistData(
                name="input_test_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                ],
            ),
            vert_woe_bin_rule,
        ],
        output_uris=[report_path],
    )

    param.inputs[0].meta.Pack(meta)
    param.inputs[1].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
        tracer_report=True,
    )

    logging.info(f"train tracer_report {res['tracer_report']}")
    comp_ret = Report()
    res = res["eval_result"]
    res.outputs[0].meta.Unpack(comp_ret)
    logging.info(comp_ret)


@pytest.mark.mpc
def test_stats_psi_woe_one_party_selected(sf_production_setup_comp):
    # the params in the following must be the same in vert_bin_rule
    alice_path = "test_io/x_alice.csv"
    bob_path = "test_io/x_bob.csv"
    report_path = "test_io/report"
    vert_woe_bin_rule = gen_vert_woe_bin_rule(sf_production_setup_comp)
    storage_config, sf_cluster_config = sf_production_setup_comp
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

    param = build_node_eval_param(
        domain="stats",
        name="stats_psi",
        version="1.0.0",
        attrs={'input/input_base_ds/feature_selects': [f"a{i}" for i in range(2)]},
        inputs=[
            DistData(
                name="input_base_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                ],
            ),
            DistData(
                name="input_test_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                ],
            ),
            vert_woe_bin_rule,
        ],
        output_uris=[report_path],
    )

    param.inputs[0].meta.Pack(meta)
    param.inputs[1].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
        tracer_report=True,
    )

    logging.info(f"train tracer_report {res['tracer_report']}")
    comp_ret = Report()
    res = res["eval_result"]
    res.outputs[0].meta.Unpack(comp_ret)
    logging.info(comp_ret)


def test_stats_psi_category():
    df = pd.DataFrame({'gender': ['M', 'F', 'F', 'M', np.nan, 'M']})
    rule_dict = {
        "name": "gender",
        "type": "category",
        "categories": ['M', 'F'],
        "filling_values": ['Male', 'Female'],
        "else_filling_value": "Unknown",
    }

    result = get_bin_counts_one_feature(rule_dict, "gender", df)

    expected = [
        ['M', 'Male', '3'],
        ['F', 'Female', '2'],
        ['nan values', 'Unknown', '1'],
    ]
    assert result == expected, "Test case for stats_psi_category failed"


def test_stats_psi_category_not_exist():
    df = pd.DataFrame({'gender': [-0.1, 2.0, 2.0, -0.1, np.nan, -0.1]})
    rule_dict = {
        "name": "gender",
        "type": "category",
        "categories": [-0.1, 2.0, -0.0796900233493871],
        "filling_values": ['Male', 'Female', 'others'],
        "else_filling_value": "Unknown",
    }

    result = get_bin_counts_one_feature(rule_dict, "gender", df)

    expected = [
        ['-0.1', 'Male', '3'],
        ['2.0', 'Female', '2'],
        ['-0.0796900233493871', 'others', '0'],
        ['nan values', 'Unknown', '1'],
    ]
    assert result == expected, "Test case for test_stats_psi_category_not_exist failed"


def test_stats_psi_0():
    base_bin_stats = [("label0", '0', '2'), ('label1', '1', '2'), ('label2', '2', '2')]
    test_bin_stats = [("label0", '0', '2'), ('label1', '1', '2'), ('label2', '2', '2')]
    psi, _ = calculate_stats_psi_one_feature(base_bin_stats, test_bin_stats)
    assert psi == 0.0


def test_stats_psi_small_enough():
    base_bin_stats = [
        ("label0", '0', '20'),
        ('label1', '1', '20'),
        ('label2', '2', '20'),
    ]
    test_bin_stats = [
        ("label0", '0', '19'),
        ('label1', '1', '21'),
        ('label2', '2', '20'),
    ]
    psi, _ = calculate_stats_psi_one_feature(base_bin_stats, test_bin_stats)
    assert psi <= good_psi_threshold


def test_stats_psi_too_large():
    base_bin_stats = [
        ("label0", '0', '20'),
        ('label1', '1', '20'),
        ('label2', '2', '20'),
    ]
    test_bin_stats = [
        ("label0", '0', '10'),
        ('label1', '1', '30'),
        ('label2', '2', '20'),
    ]
    psi, _ = calculate_stats_psi_one_feature(base_bin_stats, test_bin_stats)
    assert psi >= good_psi_threshold


def test_stats_psi_fail():
    base_bin_stats = [
        ("label0", '0', '20'),
        ('label1', '1', '20'),
        ('label2', '2', '20'),
    ]
    test_bin_stats = [("label0", '0', '20'), ('label1', '1', '20')]

    with pytest.raises(Exception):
        calculate_stats_psi_one_feature(base_bin_stats, test_bin_stats)
