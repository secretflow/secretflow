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
import uuid

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
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from secretflow.component.core import (
    DistDataType,
    build_node_eval_param,
    comp_eval,
    make_storage,
)

g_test_epoch = 3


def get_train_param(alice_path, bob_path, model_path):
    return build_node_eval_param(
        domain="ml.train",
        name="gnb_train",
        version="1.0.0",
        attrs={
            "n_classes": 3,
            "input/input_ds/label": ["y"],
            "input/input_ds/feature_selects": [f'x{i}' for i in range(100)],
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
        output_uris=[model_path],
    )


def get_pred_param(alice_path, bob_path, train_res, predict_path):
    return build_node_eval_param(
        domain="ml.predict",
        name="gnb_predict",
        version="1.0.0",
        attrs={
            "receiver": ["alice"],
            "pred_name": 'pred',
            "save_ids": False,
            'save_label': True,
            "input/input_ds/saved_features": ["x1"],
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


def get_meta_and_dump_data(sf_production_setup_comp, alice_path, bob_path):
    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)
    np.random.seed(42)
    X, y = datasets.make_blobs(n_samples=500, n_features=100, centers=3)
    feature_split = 50

    if self_party == "alice":
        ds = pd.DataFrame(
            X[:, :feature_split], columns=[f'x{i}' for i in range(feature_split)]
        )
        ds['y'] = y
        ds.to_csv(storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(
            X[:, feature_split:], columns=[f'x{i}' for i in range(feature_split, 100)]
        )
        ds.to_csv(storage.get_writer(bob_path), index=False)

    return VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 50,
                features=[f'x{i}' for i in range(50)],
                label_types=["int32"],
                labels=["y"],
            ),
            TableSchema(
                feature_types=["float32"] * 50,
                features=[f"x{i}" for i in range(50, 100)],
            ),
        ],
    )


@pytest.mark.mpc
def test_gpc(sf_production_setup_comp):
    work_path = f'test_knn_{str(uuid.uuid4())}'
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp

    train_param = get_train_param(alice_path, bob_path, model_path)
    meta = get_meta_and_dump_data(sf_production_setup_comp, alice_path, bob_path)
    train_param.inputs[0].meta.Pack(meta)

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
        tracer_report=True,
    )

    logging.info(f"train res: {train_res}")

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
            assert input_y.shape[0] == output_y.shape[0]

            logging.info(f"output y: {output_y}")
            # label & pred
            error_cnt = 0
            for i in range(input_y.shape[0]):
                if not np.array_equal(output_y['pred'][i], input_y['y'][i]):
                    error_cnt += 1
                    logging.info(
                        f"pred is not equal to y: {i}: {output_y['pred'][i]} != {input_y['y'][i]}"
                    )
            logging.info(f"error cnt: {error_cnt}, total: {input_y.shape[0]}")
            assert (error_cnt / input_y.shape[0]) < 0.1

    run_pred(predict_path, train_res['eval_result'])
