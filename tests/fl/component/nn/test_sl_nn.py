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
import os

import pandas as pd
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import (
    DistData,
    StorageConfig,
    TableSchema,
    VerticalTable,
)
from secretflow_spec.v1.report_pb2 import Report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from secretflow.component.core import DistDataType, build_node_eval_param, comp_eval
from secretflow.utils.simulation.datasets import dataset
from tests.conftest import prepare_storage_path

from .model_def import MODELS_CODE


def get_train_param(alice_path, bob_path, model_path):
    return build_node_eval_param(
        domain="ml.train",
        name="slnn_train",
        version="0.0.1",
        attrs={
            "models": MODELS_CODE,
            "epochs": 3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "validattion_prop": 0.2,
            "loss": "builtin",
            "loss/builtin": "binary_crossentropy",
            "optimizer/name": "Adam",
            "optimizer/params": "",
            "metrics": ["AUC", "Precision", "Recall", "MeanSquaredError"],
            "model_input_scheme": "tensor",
            "strategy/name": "pipeline",
            "strategy/params": '{"pipeline_size": 2}',
            "compressor/name": "topk_sparse",
            "compressor/params": '{"sparse_rate": 0.5}',
            "input/train_dataset/label": ["y"],
            "input/train_dataset/feature_selects": [
                "age",
                "job",
                "marital",
                "education",
                "default",
                "balance",
                "housing",
                "loan",
                "contact",
                "day",
                "month",
                "duration",
                "campaign",
                "pdays",
                "previous",
                "poutcome",
            ],
        },
        inputs=[
            DistData(
                name="train_dataset",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(
                        uri=alice_path,
                        party="alice",
                        format="csv",
                    ),
                    DistData.DataRef(
                        uri=bob_path,
                        party="bob",
                        format="csv",
                    ),
                ],
            ),
        ],
        output_uris=[model_path, ""],
    )


def get_pred_param(alice_path, bob_path, train_res, predict_path):
    return build_node_eval_param(
        domain="ml.predict",
        name="slnn_predict",
        version="0.0.2",
        attrs={
            "batch_size": 128,
            "receiver": ["alice"],
            "pred_name": "y_pred",
            "save_ids": False,
            "save_label": True,
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


def get_meta_and_dump_data(alice_path, bob_path, storage_config):
    local_fs_wd = storage_config.local_fs.wd

    ds = pd.read_csv(dataset('bank_marketing'), sep=';')
    encoder = LabelEncoder()
    ds['job'] = encoder.fit_transform(ds['job'])
    ds['marital'] = encoder.fit_transform(ds['marital'])
    ds['education'] = encoder.fit_transform(ds['education'])
    ds['default'] = encoder.fit_transform(ds['default'])
    ds['housing'] = encoder.fit_transform(ds['housing'])
    ds['loan'] = encoder.fit_transform(ds['loan'])
    ds['contact'] = encoder.fit_transform(ds['contact'])
    ds['poutcome'] = encoder.fit_transform(ds['poutcome'])
    ds['month'] = encoder.fit_transform(ds['month'])
    ds['y'] = encoder.fit_transform(ds['y'])

    scaler = MinMaxScaler()
    for f in ds.columns:
        ds[[f]] = scaler.fit_transform(ds[[f]])

    alice_features = ["age", "job", "marital", "education"]
    bob_features = [
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ]
    os.makedirs(
        os.path.join(local_fs_wd, "test_sl_nn"),
        exist_ok=True,
    )
    alice_data = ds[alice_features + ["y"]]
    alice_data.to_csv(os.path.join(local_fs_wd, alice_path), index=False)

    bob_data = ds[bob_features]
    bob_data.to_csv(os.path.join(local_fs_wd, bob_path), index=False)

    return VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 4,
                features=alice_features,
                labels=["y"],
                label_types=["float32"],
            ),
            TableSchema(
                feature_types=["float32"] * 12,
                features=bob_features,
            ),
        ],
    )


def test_sl_nn(sf_simulation_setup_devices):
    import os

    os.environ["ENABLE_NN"] = "true"

    alice_path = "test_sl_nn/x_alice.csv"
    bob_path = "test_sl_nn/x_bob.csv"
    model_path = "test_sl_nn/model"
    predict_path = "test_sl_nn/predict.csv"

    storage_config = StorageConfig(
        type="local_fs",
        local_fs=StorageConfig.LocalFSConfig(wd=prepare_storage_path("test")),
    )

    train_param = get_train_param(alice_path, bob_path, model_path)
    meta = get_meta_and_dump_data(alice_path, bob_path, storage_config)
    train_param.inputs[0].meta.Pack(meta)

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=None,
    )

    report = Report()
    train_res.outputs[1].meta.Unpack(report)
    logging.warning(MessageToJson(report))

    # epoch_3 auc > 0.7
    assert report.tabs[0].divs[0].children[0].table.rows[2].items[1].f > 0.7

    predict_param = get_pred_param(alice_path, bob_path, train_res, predict_path)
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = comp_eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=None,
    )

    assert len(predict_res.outputs) == 1

    input_y = pd.read_csv(os.path.join(storage_config.local_fs.wd, alice_path))
    output_y = orc.read_table(
        os.path.join(storage_config.local_fs.wd, predict_path)
    ).to_pandas()

    # label & pred
    assert output_y.shape[1] == 2

    assert input_y.shape[0] == output_y.shape[0]

    print(input_y.head(16))
    print(output_y.head(16))

    assert (
        tf.reduce_mean(tf.metrics.binary_accuracy(output_y["y"], output_y["y_pred"]))
        > 0.8
    )
