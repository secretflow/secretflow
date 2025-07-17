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
from secretflow_spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow_spec.v1.report_pb2 import Report
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from secretflow.component.core import (
    DistDataType,
    build_node_eval_param,
    comp_eval,
    make_storage,
)

g_test_epoch = 3


def get_train_param(alice_path, bob_path, model_path, report_path, checkpoint_path):
    return build_node_eval_param(
        domain="ml.train",
        name="ss_sgd_train",
        version="1.0.0",
        attrs={
            "epochs": g_test_epoch,
            "learning_rate": 0.3,
            "batch_size": 128,
            "sig_type": "t1",
            "reg_type": "logistic",
            "penalty": "l2",
            "l2_norm": 0.05,
            "report_weights": True,
            "input/input_ds/label": ["y"],
            "input/input_ds/feature_selects": [f"a{i}" for i in range(15)]
            + [f"b{i}" for i in range(15)],
        },
        inputs=[
            DistData(
                name="train_dataset",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[model_path, report_path],
        checkpoint_uri=checkpoint_path,
    )


def get_pred_param(alice_path, bob_path, train_res, predict_path):
    return build_node_eval_param(
        domain="ml.predict",
        name="ss_sgd_predict",
        version="1.0.0",
        attrs={
            "batch_size": 128,
            "receiver": ["alice"],
            "save_ids": False,
            "save_label": True,
            "input/input_ds/saved_features": ["a2", "a10"],
        },
        inputs=[
            train_res.outputs[0],
            DistData(
                name="train_dataset",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[predict_path],
    )


def get_eval_param(predict_path):
    return build_node_eval_param(
        domain="ml.eval",
        name="regression_eval",
        version="1.0.0",
        attrs={
            "bucket_size": 2,
            "input/input_ds/label": ["y"],
            "input/input_ds/prediction": ["pred"],
        },
        inputs=[
            DistData(
                name="in_ds",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=predict_path, party="alice", format="orc"),
                ],
            ),
        ],
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


@pytest.mark.parametrize("with_checkpoint", [True, False])
@pytest.mark.mpc
def test_ss_sgd(sf_production_setup_comp, with_checkpoint):
    work_path = f'test_ss_sgd_{with_checkpoint}'
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"
    report_path = f"{work_path}/model.report"
    checkpoint_path = f"{work_path}/checkpoint" if with_checkpoint else ""

    storage_config, sf_cluster_config = sf_production_setup_comp

    train_param = get_train_param(
        alice_path, bob_path, model_path, report_path, checkpoint_path
    )
    meta = get_meta_and_dump_data(sf_production_setup_comp, alice_path, bob_path)
    train_param.inputs[0].meta.Pack(meta)

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
            from pyarrow import orc

            output_y = orc.read_table(storage.get_reader(predict_path)).to_pandas()

            # label & pred
            assert output_y.shape[1] == 4

            assert input_y.shape[0] == output_y.shape[0]

            r2_score_ = r2_score(input_y["y"], output_y["pred"])

        # eval using biclassification eval
        eval_param = get_eval_param(predict_path)
        eval_meta = IndividualTable(
            schema=TableSchema(
                labels=["y", "pred"], label_types=["float32", "float32"]
            ),
        )
        eval_param.inputs[0].meta.Pack(eval_meta)

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
                r2_score_,
                comp_ret.tabs[0].divs[0].children[0].descriptions.items[0].value.f,
                decimal=5,
            )

    run_pred(predict_path, train_res)

    if with_checkpoint:
        cp_num = g_test_epoch - 1
        if 'alice' == sf_cluster_config.private_config.self_party:
            storage = make_storage(storage_config)
            for i in range(int(cp_num / 2), cp_num):
                with storage.get_writer(f"{checkpoint_path}_{i}") as f:
                    # destroy some checkpoint to rollback train progress
                    f.write(b"....")
        train_param.output_uris[0] = f"{work_path}/model.sf.2"
        train_res = comp_eval(
            param=train_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )
        run_pred(predict_path + '.2', train_res)
