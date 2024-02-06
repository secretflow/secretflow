import base64
import copy
import logging
import os
import random
import tarfile

import numpy as np
import pandas as pd
import secretflow_serving_lib as sfs
from google.protobuf import json_format
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from secretflow.component.data_utils import DistDataType
from secretflow.component.ml.boost.sgb.sgb import sgb_predict_comp, sgb_train_comp
from secretflow.component.ml.boost.ss_xgb.ss_xgb import (
    ss_xgb_predict_comp,
    ss_xgb_train_comp,
)
from secretflow.component.ml.linear.ss_glm import ss_glm_predict_comp, ss_glm_train_comp
from secretflow.component.ml.linear.ss_sgd import ss_sgd_predict_comp, ss_sgd_train_comp
from secretflow.component.model_export import model_export_comp
from secretflow.component.preprocessing.binning.vert_binning import (
    vert_bin_substitution_comp,
    vert_binning_comp,
)
from secretflow.component.preprocessing.unified_single_party_ops.feature_calculate import (
    feature_calculate,
)
from secretflow.component.preprocessing.unified_single_party_ops.onehot_encode import (
    onehot_encode,
)
from secretflow.component.preprocessing.unified_single_party_ops.substitution import (
    substitution,
)
from secretflow.spec.extend.calculate_rules_pb2 import CalculateOpRules
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from tests.conftest import TEST_STORAGE_ROOT


def eval_export(dir, comp_params, comp_res, storage_config, sf_cluster_config):
    export_model_path = os.path.join(dir, "s_model.tar.gz")

    input_datasets = []
    output_datasets = []
    component_eval_params = []

    def add_comp(param, res):
        param = copy.deepcopy(param)
        for i in param.inputs:
            input_datasets.append(json_format.MessageToJson(i, indent=0))
        for o in res.outputs:
            output_datasets.append(json_format.MessageToJson(o, indent=0))
        param.ClearField('inputs')
        param.ClearField('output_uris')
        json_param = json_format.MessageToJson(param, indent=0)
        component_eval_params.append(
            base64.b64encode(json_param.encode("utf-8")).decode("utf-8")
        )

    for p, r in zip(comp_params, comp_res):
        add_comp(p, r)

    export_param = NodeEvalParam(
        domain="model",
        name="model_export",
        version="0.0.1",
        attr_paths=[
            "model_name",
            "model_desc",
            "input_datasets",
            "output_datasets",
            "component_eval_params",
        ],
        attrs=[
            Attribute(s="test"),
            Attribute(s="test_desc"),
            Attribute(ss=input_datasets),
            Attribute(ss=output_datasets),
            Attribute(ss=component_eval_params),
        ],
        output_uris=[export_model_path],
    )

    export_res = model_export_comp.eval(
        param=export_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(export_res.outputs) == 1

    alice_tar = os.path.join(TEST_STORAGE_ROOT, "alice", export_model_path)
    bob_tar = os.path.join(TEST_STORAGE_ROOT, "bob", export_model_path)

    expected_files = {"model_file", "MANIFEST"}

    # alice
    tar_files = dict()
    with tarfile.open(alice_tar, "r:gz") as tar:
        for member in tar:
            tar_files[member.name] = tar.extractfile(member.name).read()

    assert expected_files == set(tar_files), f"alice_files {tar_files.keys()}"

    mm = json_format.Parse(tar_files["MANIFEST"], sfs.bundle_pb2.ModelManifest())
    logging.warn(f"alice MANIFEST ............ \n{mm}\n ............ \n")

    mb = sfs.bundle_pb2.ModelBundle()
    mb.ParseFromString(tar_files["model_file"])
    logging.warn(f"alice model_file ............ \n{mb}\n ............ \n")

    # bob
    tar_files = dict()
    with tarfile.open(bob_tar, "r:gz") as tar:
        for member in tar:
            tar_files[member.name] = tar.extractfile(member.name).read()

    assert expected_files == set(tar_files), f"alice_files {tar_files.keys()}"

    mm = json_format.Parse(tar_files["MANIFEST"], sfs.bundle_pb2.ModelManifest())
    logging.warn(f"bob MANIFEST ............ \n{mm}\n ............ \n")

    mb = sfs.bundle_pb2.ModelBundle()
    mb.ParseFromString(tar_files["model_file"])
    logging.warn(f"bob model_file ............ \n{mb}\n ............ \n")


def test_model_export(comp_prod_sf_cluster_config):
    alice_input_path = "test_model_export/alice.csv"
    bob_input_path = "test_model_export/bob.csv"

    bin_rule_path = "test_model_export/bin_rule"
    report_path = "test_model_export/bin_report"
    bin_output = "test_model_export/vert.csv"

    cal_output = "test_model_export/cal.csv"
    cal_rule = "test_model_export/rule.csv"
    cal_sub_output = "test_model_export/cal_sub.csv"

    onehot_encode_output = "test_model_export/onehot.csv"
    onehot_rule_path = "test_model_export/onehot.rule"
    onehot_report_path = "test_model_export/onehot.report"

    ss_glm_model_path = "test_model_export/model.sf"
    ss_glm_report_path = "test_model_export/model.report"

    ss_glm_predict_path = "test_model_export/predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

    os.makedirs(
        os.path.join(local_fs_wd, "test_model_export"),
        exist_ok=True,
    )

    def build_dataset():
        random.seed(42)
        data_len = 2048

        # f1 - f16, random data with random weight
        def _rand():
            return (random.random() - 0.5) * 2

        data = {}
        weight = [_rand() for _ in range(16)]
        for i in range(16):
            data[f"f{i+1}"] = [_rand() for _ in range(data_len)]
        # b1/b2, binning col, weight = 0.1
        data[f"b1"] = [random.random() / 2 for _ in range(data_len)]
        data[f"b2"] = [random.random() / 2 for _ in range(data_len)]
        weight.append(0.1)
        weight.append(0.1)
        data = pd.DataFrame(data)
        weight = pd.Series(weight)
        y = data.values.dot(weight)

        # o1/o2, onehot col
        def add_onehot(name, y, data):
            onehot_col = pd.Series(
                [random.choice(["A", "B", "C", "D"]) for _ in range(data_len)]
            )
            y = y + np.select(
                [
                    onehot_col == "A",
                    onehot_col == "B",
                    onehot_col == "C",
                    onehot_col == "D",
                ],
                [-0.5, -0.25, 0.25, 0.5],
            )
            data[name] = onehot_col
            return y, data

        y, data = add_onehot("o1", y, data)
        y, data = add_onehot("o2", y, data)

        y = np.select([y > 0.5, y <= 0.5], [0.0, 1.0])
        data["y"] = y

        return data

    data = build_dataset()
    if self_party == "alice":
        ds = data[[f"f{i+1}" for i in range(8)] + ["b1", "o1", "y"]]
        ds.to_csv(os.path.join(local_fs_wd, alice_input_path), index=False)

    elif self_party == "bob":
        ds = data[[f"f{i + 9}" for i in range(8)] + ["b2", "o2"]]
        ds.to_csv(os.path.join(local_fs_wd, bob_input_path), index=False)

    # binning
    bin_param = NodeEvalParam(
        domain="feature",
        name="vert_binning",
        version="0.0.2",
        attr_paths=[
            "input/input_data/feature_selects",
            "bin_num",
        ],
        attrs=[
            Attribute(ss=["b1", "b2"]),
            Attribute(i64=6),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[bin_rule_path, report_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 9 + ["str"],
                features=[f"f{i + 9}" for i in range(8)] + ["b2", "o2"],
            ),
            TableSchema(
                feature_types=["float32"] * 9 + ["str"],
                features=[f"f{i+1}" for i in range(8)] + ["b1", "o1"],
                label_types=["float32"],
                labels=["y"],
            ),
        ],
    )
    bin_param.inputs[0].meta.Pack(meta)

    bin_res = vert_binning_comp.eval(
        param=bin_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    # sub
    sub_param = NodeEvalParam(
        domain="preprocessing",
        name="vert_bin_substitution",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
        inputs=[
            bin_param.inputs[0],
            bin_res.outputs[0],
        ],
        output_uris=[bin_output],
    )

    sub_res = vert_bin_substitution_comp.eval(
        param=sub_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(sub_res.outputs) == 1

    # b1 b2 / 8

    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.UNARY
    rule.operands.extend(["+", "/", "8"])

    param = NodeEvalParam(
        domain="preprocessing",
        name="feature_calculate",
        version="0.0.1",
        attr_paths=[
            "rules",
            "input/in_ds/features",
        ],
        attrs=[
            Attribute(s=json_format.MessageToJson(rule)),
            Attribute(ss=["b1", "b2"]),
        ],
        inputs=[sub_res.outputs[0]],
        output_uris=[
            cal_output,
            cal_rule,
        ],
    )

    cal_res = feature_calculate.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    cal_sub_param = NodeEvalParam(
        domain="preprocessing",
        name="substitution",
        version="0.0.2",
        inputs=[sub_res.outputs[0], cal_res.outputs[1]],
        output_uris=[cal_sub_output],
    )

    cal_sub_res = substitution.eval(
        param=cal_sub_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    # onehot
    onehot_param = NodeEvalParam(
        domain="preprocessing",
        name="onehot_encode",
        version="0.0.2",
        attr_paths=[
            "drop_first",
            "input/input_dataset/features",
        ],
        attrs=[
            Attribute(b=False),
            Attribute(ss=["o1", "o2"]),
        ],
        # use binning sub output
        inputs=[cal_sub_res.outputs[0]],
        output_uris=[
            onehot_encode_output,
            onehot_rule_path,
            onehot_report_path,
        ],
    )

    onehot_res = onehot_encode.eval(
        param=onehot_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(onehot_res.outputs) == 3

    onehot_meta = VerticalTable()
    onehot_res.outputs[0].meta.Unpack(onehot_meta)

    all_onehot_features = []
    for s in onehot_meta.schemas:
        all_onehot_features.extend(list(s.features))

    # ss_glm
    train_param = NodeEvalParam(
        domain="ml.train",
        name="ss_glm_train",
        version="0.0.2",
        attr_paths=[
            "epochs",
            "learning_rate",
            "batch_size",
            "link_type",
            "label_dist_type",
            "optimizer",
            "l2_lambda",
            "report_weights",
            "input/train_dataset/label",
            "input/train_dataset/feature_selects",
            "input/train_dataset/offset",
            "input/train_dataset/weight",
        ],
        attrs=[
            Attribute(i64=1),
            Attribute(f=0.3),
            Attribute(i64=128),
            Attribute(s="Logit"),
            Attribute(s="Bernoulli"),
            Attribute(s="SGD"),
            Attribute(f=0.3),
            Attribute(b=True),
            Attribute(ss=["y"]),
            Attribute(ss=all_onehot_features),
            Attribute(ss=[]),
            Attribute(ss=[]),
        ],
        inputs=[onehot_res.outputs[0]],
        output_uris=[ss_glm_model_path, ss_glm_report_path],
    )

    train_res = ss_glm_train_comp.eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(train_res.outputs) == 2

    # ss glm pred

    predict_param = NodeEvalParam(
        domain="ml.predict",
        name="ss_glm_predict",
        version="0.0.1",
        attr_paths=[
            "receiver",
            "save_ids",
            "save_label",
        ],
        attrs=[
            Attribute(s="alice"),
            Attribute(b=False),
            Attribute(b=True),
        ],
        inputs=[train_res.outputs[0], onehot_res.outputs[0]],
        output_uris=[ss_glm_predict_path],
    )

    predict_res = ss_glm_predict_comp.eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    # by train comp
    eval_export(
        "test_model_export",
        [sub_param, cal_sub_param, onehot_param, train_param],
        [sub_res, cal_sub_res, onehot_res, train_res],
        storage_config,
        sf_cluster_config,
    )

    # by pred comp
    eval_export(
        "test_model_export",
        [sub_param, cal_sub_param, onehot_param, predict_param],
        [sub_res, cal_sub_res, onehot_res, predict_res],
        storage_config,
        sf_cluster_config,
    )


def get_ss_sgd_train_param(alice_path, bob_path, model_path):
    return NodeEvalParam(
        domain="ml.train",
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
            "input/train_dataset/label",
            "input/train_dataset/feature_selects",
        ],
        attrs=[
            Attribute(i64=1),
            Attribute(f=0.3),
            Attribute(i64=128),
            Attribute(s="t1"),
            Attribute(s="logistic"),
            Attribute(s="l2"),
            Attribute(f=0.05),
            Attribute(i64=2),
            Attribute(f=0.5),
            Attribute(s="policy_sgd"),
            Attribute(ss=["y"]),
            Attribute(ss=[f"a{i}" for i in range(15)] + [f"b{i}" for i in range(15)]),
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


def get_eval_param(predict_path):
    return NodeEvalParam(
        domain="ml.eval",
        name="regression_eval",
        version="0.0.1",
        attr_paths=[
            "bucket_size",
            "input/in_ds/label",
            "input/in_ds/prediction",
        ],
        attrs=[
            Attribute(i64=2),
            Attribute(ss=["y"]),
            Attribute(ss=["pred"]),
        ],
        inputs=[
            DistData(
                name="in_ds",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=predict_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[""],
    )


def get_meta_and_dump_data(dir, comp_prod_sf_cluster_config, alice_path, bob_path):
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd
    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, dir),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(os.path.join(local_fs_wd, alice_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, dir),
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


def get_pred_param(alice_path, bob_path, train_res, predict_path):
    return NodeEvalParam(
        domain="ml.predict",
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


def test_ss_sgd_export(comp_prod_sf_cluster_config):
    alice_path = "test_ss_sgd/x_alice.csv"
    bob_path = "test_ss_sgd/x_bob.csv"
    model_path = "test_ss_sgd/model.sf"
    predict_path = "test_ss_sgd/predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    train_param = get_ss_sgd_train_param(alice_path, bob_path, model_path)
    meta = get_meta_and_dump_data(
        "test_ss_sgd", comp_prod_sf_cluster_config, alice_path, bob_path
    )
    train_param.inputs[0].meta.Pack(meta)

    train_res = ss_sgd_train_comp.eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    predict_param = get_pred_param(alice_path, bob_path, train_res, predict_path)
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = ss_sgd_predict_comp.eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    # by train comp
    eval_export(
        "test_ss_sgd",
        [train_param],
        [train_res],
        storage_config,
        sf_cluster_config,
    )

    # by pred comp
    eval_export(
        "test_ss_sgd",
        [predict_param],
        [predict_res],
        storage_config,
        sf_cluster_config,
    )


def test_sgb_export(comp_prod_sf_cluster_config):
    alice_path = "test_sgb/x_alice.csv"
    bob_path = "test_sgb/x_bob.csv"

    bin_rule_path = "test_sgb/bin_rule"
    bin_output = "test_sgb/vert.csv"
    bin_report_path = "test_sgb/bin_report.json"

    model_path = "test_sgb/model.sf"
    predict_path = "test_sgb/predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    # binning
    bin_param = NodeEvalParam(
        domain="feature",
        name="vert_binning",
        version="0.0.2",
        attr_paths=[
            "input/input_data/feature_selects",
            "bin_num",
        ],
        attrs=[
            Attribute(ss=[f"a{i}" for i in range(2)] + [f"b{i}" for i in range(2)]),
            Attribute(i64=4),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[bin_rule_path, bin_report_path],
    )

    meta = get_meta_and_dump_data(
        "test_sgb", comp_prod_sf_cluster_config, alice_path, bob_path
    )
    bin_param.inputs[0].meta.Pack(meta)

    bin_res = vert_binning_comp.eval(
        param=bin_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    # sub
    sub_param = NodeEvalParam(
        domain="preprocessing",
        name="vert_bin_substitution",
        version="0.0.1",
        attr_paths=[],
        attrs=[],
        inputs=[
            bin_param.inputs[0],
            bin_res.outputs[0],
        ],
        output_uris=[bin_output],
    )

    sub_res = vert_bin_substitution_comp.eval(
        param=sub_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(sub_res.outputs) == 1

    # sgb
    train_param = NodeEvalParam(
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
            sub_res.outputs[0],
        ],
        output_uris=[model_path],
    )

    train_param.inputs[0].meta.Pack(meta)

    train_res = sgb_train_comp.eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    predict_param = NodeEvalParam(
        domain="ml.predict",
        name="sgb_predict",
        version="0.0.2",
        attr_paths=[
            "receiver",
            "save_ids",
            "save_label",
        ],
        attrs=[
            Attribute(s="alice"),
            Attribute(b=False),
            Attribute(b=True),
        ],
        inputs=[train_res.outputs[0], sub_res.outputs[0]],
        output_uris=[predict_path],
    )
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = sgb_predict_comp.eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    # by train comp
    eval_export(
        "test_sgb",
        [sub_param, train_param],
        [sub_res, train_res],
        storage_config,
        sf_cluster_config,
    )

    # by pred comp
    eval_export(
        "test_sgb",
        [sub_param, predict_param],
        [sub_res, predict_res],
        storage_config,
        sf_cluster_config,
    )


def test_ss_xgb_export(comp_prod_sf_cluster_config):
    alice_path = "test_xgb/x_alice.csv"
    bob_path = "test_xgb/x_bob.csv"
    model_path = "test_xgb/model.sf"
    predict_path = "test_xgb/predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

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

    meta = get_meta_and_dump_data(
        "test_xgb", comp_prod_sf_cluster_config, alice_path, bob_path
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
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = ss_xgb_predict_comp.eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    # by train comp
    eval_export(
        "test_xgb",
        [train_param],
        [train_res],
        storage_config,
        sf_cluster_config,
    )

    # by pred comp
    eval_export(
        "test_xgb",
        [predict_param],
        [predict_res],
        storage_config,
        sf_cluster_config,
    )
