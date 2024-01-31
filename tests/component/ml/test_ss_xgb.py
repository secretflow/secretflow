import os

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tests.conftest import TEST_STORAGE_ROOT

from secretflow.component.ml.boost.ss_xgb.ss_xgb import (
    ss_xgb_predict_comp,
    ss_xgb_train_comp,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_ss_xgb(comp_prod_sf_cluster_config):
    alice_path = "test_ss_xgb/x_alice.csv"
    bob_path = "test_ss_xgb/x_bob.csv"
    model_path = "test_ss_xgb/model.sf"
    predict_path = "test_ss_xgb/predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, "test_ss_xgb"),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(os.path.join(local_fs_wd, alice_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, "test_ss_xgb"),
            exist_ok=True,
        )

        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(os.path.join(local_fs_wd, bob_path), index=False)

    train_param = NodeEvalParam(
        domain="ml.train",
        name="ss_xgb_train",
        version="0.0.1",
        attr_paths=[
            "num_boost_round",
            "max_depth",
            "learning_rate",
            "objective",
            "reg_lambda",
            "subsample",
            "colsample_by_tree",
            "sketch_eps",
            "base_score",
            "input/train_dataset/label",
            "input/train_dataset/feature_selects",
        ],
        attrs=[
            Attribute(i64=3),
            Attribute(i64=3),
            Attribute(f=0.3),
            Attribute(s="logistic"),
            Attribute(f=0.1),
            Attribute(f=1),
            Attribute(f=1),
            Attribute(f=0.25),
            Attribute(f=0),
            Attribute(ss=["y"]),
            Attribute(ss=[f"a{i}" for i in range(15)] + [f"b{i}" for i in range(15)]),
        ],
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
        output_uris=[model_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
                label_types=["float32"],
                labels=["y"],
            ),
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    train_param.inputs[0].meta.Pack(meta)

    train_res = ss_xgb_train_comp.eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    predict_param = NodeEvalParam(
        domain="ml.predict",
        name="ss_xgb_predict",
        version="0.0.1",
        attr_paths=[
            "receiver",
            "save_ids",
            "save_label",
            "input/feature_dataset/saved_features",
        ],
        attrs=[
            Attribute(s="alice"),
            Attribute(b=False),
            Attribute(b=True),
            Attribute(ss=["a2", "a10"]),
        ],
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
    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
                label_types=["float32"],
                labels=["y"],
            ),
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = ss_xgb_predict_comp.eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    input_y = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", alice_path))
    output_y = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", predict_path))

    # label & pred
    assert output_y.shape[1] == 4

    assert input_y.shape[0] == output_y.shape[0]

    auc = roc_auc_score(input_y["y"], output_y["pred"])
    assert auc > 0.99, f"auc {auc}"
