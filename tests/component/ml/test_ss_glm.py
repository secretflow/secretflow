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
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from pyarrow import orc
from secretflow_spec.v1.report_pb2 import Report
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)


@pytest.mark.parametrize("optimizer", ["SGD", "IRLS"])
@pytest.mark.parametrize("with_checkpoint", [True, False])
@pytest.mark.parametrize("train_version", ["1.1.0", "1.0.0"])
@pytest.mark.mpc
def test_glm(sf_production_setup_comp, optimizer, with_checkpoint, train_version):
    work_path = f"test_glm_{optimizer}_{with_checkpoint}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    report_path = f"{work_path}/model.report"
    checkpoint_path = f"{work_path}/checkpoint"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        x["id1"] = pd.Series([f"{i}" for i in range(x.shape[0])])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds["id2"] = pd.Series([f"{i}" for i in range(x.shape[0])])
        ds.to_csv(storage.get_writer(bob_path), index=False)

    feature_selects = [f"a{i}" for i in range(15)] + [f"b{i}" for i in range(15)]
    train_param_dict = {
        "epochs": 10,
        "learning_rate": 0.3,
        "batch_size": 128,
        "link_type": "Logit",
        "label_dist_type": "Bernoulli",
        "optimizer": optimizer,
        "l2_lambda": 0.3,
        "infeed_batch_size_limit": 128 * 30,
        "iter_start_irls": 1,
        "exp_mode": "prime",
        "stopping_rounds": 2,
        "stopping_tolerance": 0.01,
        "stopping_metric": "RMSE",
        "report_weights": True,
        "input/input_ds/label": ["y"],
        "input/input_ds/feature_selects": feature_selects,
        # "offset": Attribute(ss=[]),
        # "weight": Attribute(ss=[]),
        "report_metric": True,
    }
    # back test old param
    if train_version == "1.0.0":
        train_param_dict["use_high_precision_exp"] = False
        train_param_dict.pop("exp_mode")

    train_param = build_node_eval_param(
        domain="ml.train",
        name="ss_glm_train",
        version=train_version,
        attrs=train_param_dict,
        inputs=[
            VTable(
                name="train_dataset",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_path,
                        party="alice",
                        format="csv",
                        ids={"id1": "str"},
                        features={f"a{i}": "float32" for i in range(15)},
                        labels={"y": "float32"},
                    ),
                    VTableParty.from_dict(
                        uri=bob_path,
                        party="bob",
                        format="csv",
                        ids={"id2": "str"},
                        features={f"b{i}": "float32" for i in range(15)},
                    ),
                ],
            ),
        ],
        output_uris=[model_path, report_path],
        checkpoint_uri=checkpoint_path if with_checkpoint else "",
    )

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
        tracer_report=True,
    )

    logging.info(f"train tracer_report {train_res['tracer_report']}")
    comp_ret = Report()
    train_res = train_res["eval_result"]
    train_res.outputs[1].meta.Unpack(comp_ret)
    logging.info(comp_ret)

    def run_pred(predict_path, train_res):
        predict_param = build_node_eval_param(
            domain="ml.predict",
            name="ss_glm_predict",
            version="1.1.0",
            attrs={
                "receiver": ["alice"],
                "save_ids": True,
                "save_label": True,
                "saved_features": ["a10", "a2"],
            },
            inputs=[train_res.outputs[0], train_param.inputs[0]],
            output_uris=[predict_path],
        )

        predict_res = comp_eval(
            param=predict_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
            tracer_report=True,
        )

        logging.info(f"predict tracer_report {predict_res['tracer_report']}")
        assert len(predict_res["eval_result"].outputs) == 1

        if "alice" == sf_cluster_config.private_config.self_party:
            storage = make_storage(storage_config)
            input_y = pd.read_csv(storage.get_reader(alice_path))
            dtype = defaultdict(np.float32)
            dtype["id1"] = np.string_
            output_y = orc.read_table(storage.get_reader(predict_path)).to_pandas()

            # label & pred
            assert output_y.shape[1] == 5

            assert set(output_y.columns) == set(["a2", "a10", "pred", "y", "id1"])

            if self_party == "alice":
                for n in ["a2", "a10", "y"]:
                    assert np.allclose(ds[n].values, output_y[n].values)
                assert np.all(ds["id1"].values == output_y["id1"].values)

            assert input_y.shape[0] == output_y.shape[0]

            auc = roc_auc_score(input_y["y"], output_y["pred"])
            assert auc > 0.99, f"auc {auc}"

    run_pred(f"{work_path}/predict.csv", train_res)

    if with_checkpoint:
        cp_num = len(comp_ret.tabs[1].divs[0].children[0].table.rows)
        if "alice" == sf_cluster_config.private_config.self_party:
            storage = make_storage(storage_config)
            for i in range(int(cp_num / 2), cp_num):
                with storage.get_writer(f"{checkpoint_path}_{i}") as f:
                    # destroy some checkpoint to rollback train progress
                    f.write(b"....")

        # run train again from checkpoint
        train_param.output_uris[0] = f"{work_path}/model.sf.2"
        train_res = comp_eval(
            param=train_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        run_pred(f"{work_path}/predict.csv.2", train_res)
