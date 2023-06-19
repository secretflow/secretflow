import os

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from secretflow.component.data_utils import DistDataType
from secretflow.component.ml.linear.ss_sgd import ss_sgd_predict_comp, ss_sgd_train_comp
from secretflow.protos.component.comp_pb2 import Attribute
from secretflow.protos.component.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.protos.component.evaluation_pb2 import NodeEvalParam
from tests.conftest import TEST_STORAGE_ROOT


def test_ss_sgd(comp_prod_sf_cluster_config):
    alice_path = "test_ss_sgd/x_alice.csv"
    bob_path = "test_ss_sgd/x_bob.csv"
    model_path = "test_ss_sgd/model.sf"
    predict_path = "test_ss_sgd/predict.csv"

    self_party = comp_prod_sf_cluster_config.private_config.self_party
    local_fs_wd = comp_prod_sf_cluster_config.private_config.storage_config.local_fs.wd

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, "test_ss_sgd"),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f'a{i}' for i in range(15)])
        y = pd.DataFrame(y, columns=['y'])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(os.path.join(local_fs_wd, alice_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, "test_ss_sgd"),
            exist_ok=True,
        )

        ds = pd.DataFrame(x[:, 15:], columns=[f'b{i}' for i in range(15)])
        ds.to_csv(os.path.join(local_fs_wd, bob_path), index=False)

    train_param = NodeEvalParam(
        domain="ml.linear",
        name="ss_sgd_train",
        version="0.0.1",
        attr_paths=[
            "epochs",
            "learning_rate",
            "batch_size",
            "sig_type",
            "reg_type",
            "penalty",
            "l2_norm",
            "decay_epoch",
            "decay_rate",
            "strategy",
        ],
        attrs=[
            Attribute(i64=3),
            Attribute(f=0.3),
            Attribute(i64=128),
            Attribute(s="t1"),
            Attribute(s="logistic"),
            Attribute(s="l2"),
            Attribute(f=0.05),
            Attribute(i64=2),
            Attribute(f=0.5),
            Attribute(s="policy_sgd"),
        ],
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

    meta = VerticalTable(
        schemas=[
            TableSchema(
                types=["f32"] * 15,
                features=[f"a{i}" for i in range(15)],
                labels=["y"],
            ),
            TableSchema(
                types=["f32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    train_param.inputs[0].meta.Pack(meta)

    train_res = ss_sgd_train_comp.eval(train_param, comp_prod_sf_cluster_config)

    predict_param = NodeEvalParam(
        domain="ml.linear",
        name="ss_sgd_predict",
        version="0.0.1",
        attr_paths=[
            "batch_size",
            "receiver",
            "save_ids",
            "save_label",
        ],
        attrs=[
            Attribute(i64=128),
            Attribute(s="alice"),
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
    meta = VerticalTable(
        schemas=[
            TableSchema(
                types=["f32"] * 15,
                features=[f"a{i}" for i in range(15)],
                labels=["y"],
            ),
            TableSchema(
                types=["f32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = ss_sgd_predict_comp.eval(predict_param, comp_prod_sf_cluster_config)

    assert len(predict_res.outputs) == 1

    input_y = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", alice_path))
    output_y = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", predict_path))

    # label & pred
    assert output_y.shape[1] == 2

    assert input_y.shape[0] == output_y.shape[0]

    auc = roc_auc_score(input_y["y"], output_y["pred"])
    assert auc > 0.99, f"auc {auc}"
