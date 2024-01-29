import logging
import os

import numpy as np
import pandas as pd
from google.protobuf.json_format import MessageToJson
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from secretflow.component.ml.boost.sgb.sgb import sgb_predict_comp, sgb_train_comp
from secretflow.component.ml.eval.biclassification_eval import (
    biclassification_eval_comp,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report
from tests.conftest import TEST_STORAGE_ROOT


def get_train_param(alice_path, bob_path, model_path):
    return NodeEvalParam(
        domain="ml.train",
        name="sgb_train",
        version="0.0.1",
        attr_paths=[
            "num_boost_round",
            "max_depth",
            "learning_rate",
            "objective",
            "reg_lambda",
            "gamma",
            "rowsample_by_tree",
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
            Attribute(f=0),
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


def get_pred_param(alice_path, bob_path, train_res, predict_path):
    return NodeEvalParam(
        domain="ml.predict",
        name="sgb_predict",
        version="0.0.2",
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
            Attribute(ss=[f"a12", "a1", "a6"]),
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


def get_eval_param(input_dd):
    return NodeEvalParam(
        domain="ml.eval",
        name="biclassification_eval",
        version="0.0.1",
        attr_paths=[
            "bucket_size",
            "min_item_cnt_per_bucket",
            "input/in_ds/label",
            "input/in_ds/prediction",
        ],
        attrs=[
            Attribute(i64=2),
            Attribute(i64=5),
            Attribute(ss=["y"]),
            Attribute(ss=["pred"]),
        ],
        inputs=[input_dd],
        output_uris=[""],
    )


def get_meta_and_dump_data(comp_prod_sf_cluster_config, alice_path, bob_path):
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd
    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, "test_sgb"),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(os.path.join(local_fs_wd, alice_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, "test_sgb"),
            exist_ok=True,
        )

        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(os.path.join(local_fs_wd, bob_path), index=False)

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


def test_sgb(comp_prod_sf_cluster_config):
    alice_path = "test_sgb/x_alice.csv"
    bob_path = "test_sgb/x_bob.csv"
    model_path = "test_sgb/model.sf"
    predict_path = "test_sgb/predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    train_param = get_train_param(alice_path, bob_path, model_path)
    meta = get_meta_and_dump_data(comp_prod_sf_cluster_config, alice_path, bob_path)
    train_param.inputs[0].meta.Pack(meta)

    train_res = sgb_train_comp.eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    predict_param = get_pred_param(alice_path, bob_path, train_res, predict_path)
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = sgb_predict_comp.eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    input_y = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", alice_path))
    output_y = pd.read_csv(os.path.join(TEST_STORAGE_ROOT, "alice", predict_path))

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
    assert auc > 0.99, f"auc {auc}"

    output_it = IndividualTable()

    assert predict_res.outputs[0].meta.Unpack(output_it)
    assert output_it.line_count == input_y.shape[0]

    logging.warning(f"pred .......")

    # eval using biclassification eval
    eval_param = get_eval_param(predict_res.outputs[0])

    eval_res = biclassification_eval_comp.eval(
        param=eval_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    comp_ret = Report()
    eval_res.outputs[0].meta.Unpack(comp_ret)
    logging.warning(MessageToJson(comp_ret))
    np.testing.assert_almost_equal(
        auc,
        comp_ret.tabs[0].divs[0].children[0].descriptions.items[3].value.f,
        decimal=2,
    )
