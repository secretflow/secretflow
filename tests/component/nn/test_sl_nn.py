import logging
import os

import pandas as pd
import tensorflow as tf
from google.protobuf.json_format import MessageToJson
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from secretflow.component.data_utils import DistDataType
from secretflow.component.ml.nn.sl import slnn_predict_comp, slnn_train_comp
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    StorageConfig,
    TableSchema,
    VerticalTable,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report
from secretflow.utils.simulation.datasets import dataset
from tests.conftest import prepare_storage_path

from .model_def import MODELS_CODE


def get_train_param(alice_path, bob_path, model_path):
    return NodeEvalParam(
        domain="ml.train",
        name="slnn_train",
        version="0.0.1",
        attr_paths=[
            "models",
            "epochs",
            "learning_rate",
            "batch_size",
            "validattion_prop",
            "loss/builtin",
            "optimizer/name",
            "optimizer/params",
            "metrics",
            "model_input_scheme",
            "strategy/name",
            "strategy/params",
            "compressor/name",
            "compressor/params",
            "input/train_dataset/label",
            "input/train_dataset/feature_selects",
        ],
        attrs=[
            Attribute(s=MODELS_CODE),
            Attribute(i64=3),
            Attribute(f=0.001),
            Attribute(i64=32),
            Attribute(f=0.2),
            Attribute(s="binary_crossentropy"),
            Attribute(s="Adam"),
            Attribute(s=""),
            Attribute(
                ss=[
                    "AUC",
                    "Precision",
                    "Recall",
                    "MeanSquaredError",
                ]
            ),
            Attribute(s="tensor"),
            Attribute(s="pipeline"),
            Attribute(s='{"pipeline_size": 2}'),
            Attribute(s="topk_sparse"),
            Attribute(s='{"sparse_rate": 0.5}'),
            Attribute(ss=["y"]),
            Attribute(
                ss=[
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
                ]
            ),
        ],
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
    return NodeEvalParam(
        domain="ml.predict",
        name="slnn_predict",
        version="0.0.1",
        attr_paths=[
            "batch_size",
            "receiver",
            "pred_name",
            "save_ids",
            "save_label",
        ],
        attrs=[
            Attribute(i64=128),
            Attribute(s="alice"),
            Attribute(s="y_pred"),
            Attribute(b=False),
            Attribute(b=True),
        ],
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

    train_res = slnn_train_comp.eval(
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

    predict_res = slnn_predict_comp.eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=None,
    )

    assert len(predict_res.outputs) == 1

    input_y = pd.read_csv(os.path.join(storage_config.local_fs.wd, alice_path))
    output_y = pd.read_csv(os.path.join(storage_config.local_fs.wd, predict_path))

    # label & pred
    assert output_y.shape[1] == 2

    assert input_y.shape[0] == output_y.shape[0]

    print(input_y.head(16))
    print(output_y.head(16))

    assert (
        tf.reduce_mean(tf.metrics.binary_accuracy(output_y["y"], output_y["y_pred"]))
        > 0.8
    )
