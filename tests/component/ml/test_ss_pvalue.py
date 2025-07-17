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

import pandas as pd
import pytest
from secretflow_spec.v1.report_pb2 import Report
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


@pytest.mark.parametrize("reg_type", ["logistic", "linear"])
@pytest.mark.mpc
def test_ss_pvalue(sf_production_setup_comp, reg_type):
    alice_input_path = f"test_ss_pvalue{reg_type}/alice.csv"
    bob_input_path = f"test_ss_pvalue{reg_type}/bob.csv"
    model_path = f"test_ss_pvalue{reg_type}/model.sf"
    report_path = f"test_ss_pvalue{reg_type}/model.report"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(storage.get_writer(alice_input_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(storage.get_writer(bob_input_path), index=False)

    train_param = build_node_eval_param(
        domain="ml.train",
        name="ss_sgd_train",
        version="1.0.0",
        attrs={
            "epochs": 3,
            "learning_rate": 0.3,
            "batch_size": 128,
            "sig_type": "t1",
            "reg_type": reg_type,
            "penalty": "l2",
            "l2_norm": 0.05,
            "input/input_ds/label": ["y"],
            "input/input_ds/feature_selects": [f"b{i}" for i in range(15)],
        },
        inputs=[
            VTable(
                name="train_dataset",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                        features={f"a{i}": "float32" for i in range(15)},
                        labels={"y": "float32"},
                    ),
                    VTableParty.from_dict(
                        uri=bob_input_path,
                        party="bob",
                        format="csv",
                        features={f"b{i}": "float32" for i in range(15)},
                    ),
                ],
            ),
        ],
        output_uris=[model_path, report_path],
    )
    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    pv_param = build_node_eval_param(
        domain="ml.eval",
        name="ss_pvalue",
        version="1.0.0",
        attrs=None,
        inputs=[train_res.outputs[0], train_param.inputs[0]],
        output_uris=["report"],
    )

    res = comp_eval(
        param=pv_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    report = Report()
    assert res.outputs[0].meta.Unpack(report)

    logging.info(report)

    assert len(report.tabs) == 1
    tab = report.tabs[0]
    assert len(tab.divs) == 1
    div = tab.divs[0]
    assert len(div.children) == 1
    c = div.children[0]
    assert c.type == "descriptions"
    descriptions = c.descriptions
    assert len(descriptions.items) == 15 + 1


@pytest.mark.mpc
def test_ss_pvalue_glm(sf_production_setup_comp):
    alice_input_path = f"test_ss_pvalue_glm/alice.csv"
    bob_input_path = f"test_ss_pvalue_glm/bob.csv"
    model_path = f"test_ss_pvalue_glm/model.sf"
    report_path = f"test_ss_pvalue_glm/report.sf"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(storage.get_writer(alice_input_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(storage.get_writer(bob_input_path), index=False)

    train_param = build_node_eval_param(
        domain="ml.train",
        name="ss_glm_train",
        version="1.0.0",
        attrs={
            "epochs": 10,
            "learning_rate": 0.3,
            "batch_size": 128,
            "link_type": "Logit",
            "label_dist_type": "Bernoulli",
            "optimizer": "IRLS",
            "l2_lambda": 0.3,
            "infeed_batch_size_limit": 128 * 30,
            "iter_start_irls": 1,
            "stopping_rounds": 2,
            "stopping_tolerance": 0.01,
            "stopping_metric": "RMSE",
            "report_weights": True,
            "input/input_ds/label": ["y"],
            "input/input_ds/feature_selects": [f"a{i}" for i in range(15)]
            + [f"b{i}" for i in range(15)],
            # "offset": Attribute(ss=[]),
            # "weight": Attribute(ss=[]),
            "report_metric": True,
        },
        inputs=[
            VTable(
                name="train_dataset",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                        features={f"a{i}": "float32" for i in range(15)},
                        labels={"y": "float32"},
                    ),
                    VTableParty.from_dict(
                        uri=bob_input_path,
                        party="bob",
                        format="csv",
                        features={f"b{i}": "float32" for i in range(15)},
                    ),
                ],
            ),
        ],
        output_uris=[model_path, report_path],
    )

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    pv_param = build_node_eval_param(
        domain="ml.eval",
        name="ss_pvalue",
        version="1.0.0",
        attrs=None,
        inputs=[train_res.outputs[0], train_param.inputs[0]],
        output_uris=["report"],
    )

    res = comp_eval(
        param=pv_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 1

    report = Report()
    assert res.outputs[0].meta.Unpack(report)

    logging.info(report)

    assert len(report.tabs) == 1
    tab = report.tabs[0]
    assert len(tab.divs) == 1
    div = tab.divs[0]
    assert len(div.children) == 1
    c = div.children[0]
    assert c.type == "descriptions"
    descriptions = c.descriptions
    assert len(descriptions.items) == 30 + 1
