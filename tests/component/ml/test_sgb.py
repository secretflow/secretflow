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
from google.protobuf.json_format import MessageToJson
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow_spec.v1.report_pb2 import Report
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from secretflow.component.core import build_node_eval_param, comp_eval, make_storage

NUM_BOOST_ROUND = 5


OBJECTIVE_CASES = ["biclassification", "tweedie_regression"]


def get_train_param(
    alice_path,
    bob_path,
    report_path,
    model_path,
    checkpoint_uri,
    train_version,
    objective_case='biclassification',
):
    objective = "logistic"
    metric = "roc_auc"
    if objective_case == "tweedie_regression":
        objective = "tweedie"
        metric = "tweedie_deviance"
    params = {
        "num_boost_round": NUM_BOOST_ROUND,
        "max_depth": 3,
        "learning_rate": 0.3,
        "objective": objective,
        "reg_lambda": 0.1,
        "gamma": 0.0,
        "rowsample_by_tree": 1.0,
        "colsample_by_tree": 1.0,
        "sketch_eps": 0.25,
        "base_score": 0.0,
        "enable_monitor": True,
        "enable_early_stop": False if checkpoint_uri else True,
        "eval_metric": metric,
        "validation_fraction": 0.1,
        "stopping_rounds": 3,
        "stopping_tolerance": 0.01,
        "save_best_model": True,
        "tweedie_variance_power": 1.65,
        "input/input_ds/label": ["y"],
        "input/input_ds/feature_selects": [f"a{i}" for i in range(15)]
        + [f"b{i}" for i in range(15)],
    }
    outputs = [model_path]
    if train_version >= "1.1.0":
        params["report_importances"] = True
        outputs.append(report_path)
    return build_node_eval_param(
        domain="ml.train",
        name="sgb_train",
        version=train_version,
        attrs=params,
        inputs=[
            DistData(
                name="train_dataset",
                type="sf.table.vertical_table",
                data_refs=[
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=outputs,
        checkpoint_uri=checkpoint_uri,
    )


def get_pred_param(alice_path, bob_path, train_res, predict_path):
    return build_node_eval_param(
        domain="ml.predict",
        name="sgb_predict",
        version="1.0.0",
        attrs={
            "receiver": ["alice"],
            "save_ids": False,
            "save_label": True,
            "saved_features": ["a12", "a1", "a6"],
        },
        inputs=[
            train_res.outputs[0],
            DistData(
                name="train_dataset",
                type="sf.table.vertical_table",
                data_refs=[
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[predict_path],
    )


def get_eval_param(input_dd):
    return build_node_eval_param(
        domain="ml.eval",
        name="biclassification_eval",
        version="1.0.0",
        attrs={
            "bucket_size": 2,
            "min_item_cnt_per_bucket": 5,
            "input/input_ds/label": ["y"],
            "input/input_ds/prediction": ["pred"],
        },
        inputs=[input_dd],
        output_uris=[""],
    )


def get_meta_and_dump_data(sf_production_setup_comp, alice_path, bob_path):
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
        ds.to_csv(storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(storage.get_writer(bob_path), index=False)

    return VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
                labels=["y"],
                label_types=["float32"],
            ),
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )


def _inner_test_sgb(
    sf_production_setup_comp, with_checkpoint, objective_case, train_version
):
    work_path = f"test_sgb_{with_checkpoint}_{objective_case}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    report_path = f"{work_path}/model.report"
    predict_path = f"{work_path}/predict.csv"
    checkpoint_path = f"{work_path}/checkpoint"

    storage_config, sf_cluster_config = sf_production_setup_comp

    train_param = get_train_param(
        alice_path,
        bob_path,
        report_path,
        model_path,
        checkpoint_path if with_checkpoint else "",
        train_version,
        objective_case,
    )
    meta = get_meta_and_dump_data(sf_production_setup_comp, alice_path, bob_path)
    train_param.inputs[0].meta.Pack(meta)

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
        tracer_report=True if train_version >= "1.1.0" else False,
    )

    if train_version >= "1.1.0":
        comp_ret = Report()
        train_res = train_res["eval_result"]
        train_res.outputs[1].meta.Unpack(comp_ret)
        logging.info(comp_ret)

    def run_pred(predict_path, train_res):
        predict_param = get_pred_param(alice_path, bob_path, train_res, predict_path)
        predict_param.inputs[1].meta.Pack(meta)

        predict_res = comp_eval(
            param=predict_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        assert len(predict_res.outputs) == 1

        if "alice" == sf_cluster_config.private_config.self_party:
            storage = make_storage(storage_config)
            input_y = pd.read_csv(storage.get_reader(alice_path))
            output_y = orc.read_table(storage.get_reader(predict_path)).to_pandas()

            output_it = IndividualTable()

            assert predict_res.outputs[0].meta.Unpack(output_it)
            assert output_it.line_count == input_y.shape[0]

            # label & pred
            assert output_y.shape[1] == 5

            np.testing.assert_almost_equal(
                input_y["a1"].values, output_y["a1"].values, decimal=4
            )
            np.testing.assert_almost_equal(
                input_y["a6"].values, output_y["a6"].values, decimal=4
            )
            np.testing.assert_almost_equal(
                input_y["a12"].values, output_y["a12"].values, decimal=4
            )

            assert input_y.shape[0] == output_y.shape[0]

            auc = roc_auc_score(input_y["y"], output_y["pred"])
            if objective_case == "biclassification":
                assert auc > 0.98, f"auc {auc}"
            # note that this is not a good test for tweedie
            # see tests/ml/boost/sgb_v/test_vert_sgb_with_xgb.py for reference
            # for a good tweedie test
            elif objective_case == "tweedie_regression":
                assert auc > 0.95, f"auc {auc}"

            output_it = IndividualTable()

            assert predict_res.outputs[0].meta.Unpack(output_it)
            assert output_it.line_count == input_y.shape[0]

        logging.warning(f"pred .......")

        # eval using biclassification eval
        eval_param = get_eval_param(predict_res.outputs[0])

        eval_res = comp_eval(
            param=eval_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )
        comp_ret = Report()
        eval_res.outputs[0].meta.Unpack(comp_ret)
        logging.warning(MessageToJson(comp_ret))
        if "alice" == sf_cluster_config.private_config.self_party:
            np.testing.assert_almost_equal(
                auc,
                comp_ret.tabs[0].divs[0].children[0].descriptions.items[3].value.f,
                decimal=2,
            )

    run_pred(predict_path, train_res)
    if with_checkpoint:
        cp_num = NUM_BOOST_ROUND
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


@pytest.mark.parametrize("objective_case", OBJECTIVE_CASES)
@pytest.mark.parametrize("train_version", ["1.1.0", "1.0.0"])
@pytest.mark.mpc
def test_sgb_with_checkpoint(sf_production_setup_comp, objective_case, train_version):
    _inner_test_sgb(sf_production_setup_comp, True, objective_case, train_version)


@pytest.mark.parametrize("objective_case", OBJECTIVE_CASES)
@pytest.mark.parametrize("train_version", ["1.0.0", "1.1.0"])
@pytest.mark.mpc
def test_sgb_without_checkpoint(
    sf_production_setup_comp, objective_case, train_version
):
    _inner_test_sgb(sf_production_setup_comp, False, objective_case, train_version)
