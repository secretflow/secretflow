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

import random

import numpy as np
import pandas as pd
import pytest
from google.protobuf import json_format
from secretflow_spec.v1.data_pb2 import DistData, TableSchema, VerticalTable

from secretflow.component.core import (
    DistDataType,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.spec.extend.calculate_rules_pb2 import CalculateOpRules
from tests.component.infra.util import (
    eval_export,
    get_meta_and_dump_data,
    get_pred_param,
    get_ss_sgd_train_param,
    setup_cluster_config,
)


@pytest.mark.parametrize(
    "features_in_one_party, he_mode",
    [(False, True), (False, False), (True, True), (True, False)],
)
@pytest.mark.mpc
def test_ss_sgd_export(sf_production_setup_comp, features_in_one_party, he_mode):
    work_path = f"test_ss_sgd_{features_in_one_party}_{he_mode}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    report_path = f"{work_path}/model.report"
    predict_path = f"{work_path}/predict.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    sf_cluster_config = setup_cluster_config(sf_cluster_config, he_mode)

    train_param = get_ss_sgd_train_param(alice_path, bob_path, model_path, report_path)
    meta = get_meta_and_dump_data(
        work_path,
        sf_production_setup_comp,
        alice_path,
        bob_path,
        features_in_one_party,
    )
    train_param.inputs[0].meta.Pack(meta)

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    predict_param = get_pred_param(alice_path, bob_path, train_res, predict_path)
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = comp_eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    expected_input = [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]

    # by train comp
    eval_export(
        work_path,
        [train_param],
        [train_res],
        storage_config,
        sf_cluster_config,
        expected_input,
        he_mode,
    )

    # by pred comp
    eval_export(
        work_path,
        [predict_param],
        [predict_res],
        storage_config,
        sf_cluster_config,
        expected_input,
        he_mode,
    )


@pytest.mark.parametrize("features_in_one_party", [True, False])
@pytest.mark.parametrize("he_mode", [True, False])
@pytest.mark.mpc
def test_ss_xgb_export(sf_production_setup_comp, features_in_one_party, he_mode):
    work_path = f"test_ss_xgb_{features_in_one_party}_{he_mode}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    sf_cluster_config = setup_cluster_config(sf_cluster_config, he_mode)

    train_param = build_node_eval_param(
        domain="ml.train",
        name="ss_xgb_train",
        version="1.0.0",
        attrs={
            "num_boost_round": 2,
            "max_depth": 2,
            "learning_rate": 0.3,
            "objective": "logistic",
            "reg_lambda": 0.1,
            "subsample": 1.0,
            "colsample_by_tree": 1.0,
            "sketch_eps": 0.25,
            "base_score": 0.0,
            "input/input_ds/label": ["y"],
            "input/input_ds/feature_selects": [f"a{i}" for i in range(4)]
            + [f"b{i}" for i in range(4)],
        },
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
        work_path,
        sf_production_setup_comp,
        alice_path,
        bob_path,
        features_in_one_party,
    )
    train_param.inputs[0].meta.Pack(meta)

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    predict_param = build_node_eval_param(
        domain="ml.predict",
        name="ss_xgb_predict",
        version="1.0.0",
        attrs={
            "receiver": ["alice"],
            "save_ids": False,
            "save_label": True,
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
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = comp_eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    expected_input = [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]

    # by train comp
    eval_export(
        work_path,
        [train_param],
        [train_res],
        storage_config,
        sf_cluster_config,
        expected_input,
        he_mode,
    )

    # by pred comp
    eval_export(
        work_path,
        [predict_param],
        [predict_res],
        storage_config,
        sf_cluster_config,
        expected_input,
        he_mode,
    )


@pytest.mark.mpc
def test_score_card_transformer_export(sf_production_setup_comp):
    work_path = "test_score_card_transformer_export"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"
    score_card_trans_path = f"{work_path}/score_card_trans.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp

    train_param = build_node_eval_param(
        domain="ml.train",
        name="ss_xgb_train",
        version="1.0.0",
        attrs={
            "num_boost_round": 3,
            "max_depth": 3,
            "learning_rate": 0.3,
            "objective": "logistic",
            "reg_lambda": 0.1,
            "subsample": 1.0,
            "colsample_by_tree": 1.0,
            "sketch_eps": 0.25,
            "base_score": 0.0,
            "label": ["y"],
            "feature_selects": [f"a{i}" for i in range(4)]
            + [f"b{i}" for i in range(4)],
        },
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
        work_path,
        sf_production_setup_comp,
        alice_path,
        bob_path,
        False,
    )
    train_param.inputs[0].meta.Pack(meta)

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    predict_param = build_node_eval_param(
        domain="ml.predict",
        name="ss_xgb_predict",
        version="1.0.0",
        attrs={
            "receiver": ["alice"],
            "save_ids": False,
            "save_label": True,
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
    predict_param.inputs[1].meta.Pack(meta)

    predict_res = comp_eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    # score_card_transformer
    score_card_trans_param = build_node_eval_param(
        domain="postprocessing",
        name="score_card_transformer",
        version="1.0.0",
        attrs={
            "positive": 1,
            "predict_score_name": "predict_score",
            "scaled_value": 600,
            "odd_base": 20.0,
            "pdo": 20.0,
            "input/input_ds/predict_name": ["pred"],
        },
        inputs=[predict_res.outputs[0]],
        output_uris=[score_card_trans_path],
    )

    score_card_trans_res = comp_eval(
        param=score_card_trans_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    expected_input = [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]

    eval_export(
        work_path,
        [predict_param, score_card_trans_param],
        [predict_res, score_card_trans_res],
        storage_config,
        sf_cluster_config,
        expected_input,
    )


def _inner_test_model_export(sf_production_setup_comp, features_in_one_party, he_mode):
    work_path = f"test_model_export_{features_in_one_party}"
    alice_input_path = f"{work_path}/alice.csv"
    bob_input_path = f"{work_path}/bob.csv"

    bin_rule_path = f"{work_path}/bin_rule"
    report_path = f"{work_path}/bin_report"
    bin_output = f"{work_path}/vert.csv"

    cal_output = f"{work_path}/cal.csv"
    cal_rule = f"{work_path}/rule.csv"
    cal_sub_output = f"{work_path}/cal_sub.csv"

    onehot_encode_output = f"{work_path}/onehot.csv"
    onehot_rule_path = f"{work_path}/onehot.rule"
    onehot_report_path = f"{work_path}/onehot.report"

    ss_glm_model_path = f"{work_path}/model.sf"
    ss_glm_report_path = f"{work_path}/model.report"

    ss_glm_predict_path = f"{work_path}/predict.csv"

    score_card_trans_path = f"{work_path}/score_card_transformer.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp
    sf_cluster_config = setup_cluster_config(sf_cluster_config, he_mode)
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    def build_dataset():
        random.seed(42)
        data_len = 32

        # f1 - f8, random data with random weight
        def _rand():
            return (random.random() - 0.5) * 2

        data = {}
        weight = [_rand() for _ in range(8)]
        for i in range(8):
            data[f"f{i+1}"] = [_rand() for _ in range(data_len)]

        # b1/b2, binning col, weight = 0.1
        data[f"b1"] = [random.random() / 2 for _ in range(data_len)]
        data[f"b2"] = [random.random() / 2 for _ in range(data_len)]
        weight.append(0.1)
        weight.append(0.1)
        weight = pd.Series(weight)
        y = pd.DataFrame(data).values.dot(weight)

        # unused1 / unused2 is unused.... test input trace
        for i in range(2):
            data[f"unused{i+1}"] = [_rand() for _ in range(data_len)]

        data = pd.DataFrame(data)

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
        if features_in_one_party:
            #  alice has y
            ds = data[["y"]]
            ds.to_csv(storage.get_writer(alice_input_path), index=False)
        else:
            ds = data[[f"f{i+1}" for i in range(4)] + ["b1", "o1", "y", "unused1"]]
            ds.to_csv(storage.get_writer(alice_input_path), index=False)

    elif self_party == "bob":
        if features_in_one_party:
            #  bob has all features
            ds = data[
                [f"f{i + 1}" for i in range(8)]
                + ["b1", "b2", "o1", "o2", "unused1", "unused2"]
            ]
            ds.to_csv(storage.get_writer(bob_input_path), index=False)
        else:
            ds = data[[f"f{i + 5}" for i in range(4)] + ["b2", "o2", "unused2"]]
            ds.to_csv(storage.get_writer(bob_input_path), index=False)

    # binning
    bin_param = build_node_eval_param(
        domain="preprocessing",
        name="vert_binning",
        version="1.0.0",
        attrs={
            "input/input_ds/feature_selects": ["b1", "b2"],
            "bin_num": 6,
        },
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
        output_uris=[bin_output, bin_rule_path, report_path],
    )

    if features_in_one_party:
        meta = VerticalTable(
            schemas=[
                TableSchema(
                    feature_types=["float32"] * 12 + ["str"] * 2,
                    features=[f"f{i + 1}" for i in range(8)]
                    + ["unused1", "unused2", "b1", "b2", "o1", "o2"],
                ),
                TableSchema(
                    feature_types=[],
                    features=[],
                    label_types=["float32"],
                    labels=["y"],
                ),
            ],
        )
    else:
        meta = VerticalTable(
            schemas=[
                TableSchema(
                    feature_types=["float32"] * 6 + ["str"],
                    features=[f"f{i + 5}" for i in range(4)] + ["unused2", "b2", "o2"],
                ),
                TableSchema(
                    feature_types=["float32"] * 6 + ["str"],
                    features=[f"f{i+1}" for i in range(4)] + ["unused1", "b1", "o1"],
                    label_types=["float32"],
                    labels=["y"],
                ),
            ],
        )
    bin_param.inputs[0].meta.Pack(meta)

    bin_res = comp_eval(
        param=bin_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    # sub
    sub_param = build_node_eval_param(
        domain="preprocessing",
        name="substitution",
        version="1.0.0",
        attrs=None,
        inputs=[bin_param.inputs[0], bin_res.outputs[1]],
        output_uris=[bin_output],
    )

    sub_res = comp_eval(
        param=sub_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(sub_res.outputs) == 1

    # b1 b2 / 8

    rule = CalculateOpRules()
    rule.op = CalculateOpRules.OpType.UNARY
    rule.operands.extend(["+", "/", "8"])

    param = build_node_eval_param(
        domain="preprocessing",
        name="feature_calculate",
        version="1.0.0",
        attrs={
            "rules": json_format.MessageToJson(rule),
            "input/input_ds/features": ["b1", "b2"],
        },
        inputs=[sub_res.outputs[0]],
        output_uris=[cal_output, cal_rule],
    )

    cal_res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    cal_sub_param = build_node_eval_param(
        domain="preprocessing",
        name="substitution",
        version="1.0.0",
        attrs=None,
        inputs=[sub_res.outputs[0], cal_res.outputs[1]],
        output_uris=[cal_sub_output],
    )

    cal_sub_res = comp_eval(
        param=cal_sub_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    # onehot
    onehot_param = build_node_eval_param(
        domain="preprocessing",
        name="onehot_encode",
        version="1.0.0",
        attrs={
            "drop": "no_drop",
            "input/input_ds/features": ["o1", "o2"],
        },
        # use binning sub output
        inputs=[cal_sub_res.outputs[0]],
        output_uris=[
            onehot_encode_output,
            onehot_rule_path,
            onehot_report_path,
        ],
    )

    onehot_res = comp_eval(
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

    all_onehot_features.remove("unused1")
    all_onehot_features.remove("unused2")
    all_onehot_features.remove("f1")
    all_onehot_features.remove("o1_1")

    # ss_glm
    train_param = build_node_eval_param(
        domain="ml.train",
        name="ss_glm_train",
        version="1.0.0",
        attrs={
            "epochs": 1,
            "learning_rate": 0.3,
            "batch_size": 32,
            "link_type": "Logit",
            "label_dist_type": "Bernoulli",
            "optimizer": "SGD",
            "l2_lambda": 0.3,
            "report_weights": True,
            "input/input_ds/label": ["y"],
            "input/input_ds/feature_selects": all_onehot_features,
            "input/input_ds/offset": ["f1"],
            # "input/input_ds/weight": Attribute(ss=[]),
        },
        inputs=[onehot_res.outputs[0]],
        output_uris=[ss_glm_model_path, ss_glm_report_path],
    )

    expected_input = [f"f{i + 1}" for i in range(8)] + ["b1", "b2", "o1", "o2"]

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(train_res.outputs) == 2

    # ss glm pred
    predict_param = build_node_eval_param(
        domain="ml.predict",
        name="ss_glm_predict",
        version="1.0.0",
        attrs={
            "receiver": ["alice"],
            "save_ids": False,
            "save_label": True,
        },
        inputs=[train_res.outputs[0], onehot_res.outputs[0]],
        output_uris=[ss_glm_predict_path],
    )

    predict_res = comp_eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    score_card_trans_param = build_node_eval_param(
        domain="postprocessing",
        name="score_card_transformer",
        version="1.0.0",
        attrs={
            "positive": 1,
            "predict_score_name": "predict_score",
            "scaled_value": 600,
            "odd_base": 20.0,
            "pdo": 20.0,
            "input/input_ds/predict_name": ["pred"],
        },
        inputs=[predict_res.outputs[0]],
        output_uris=[score_card_trans_path],
    )

    score_card_trans_res = comp_eval(
        param=score_card_trans_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    # by train comp
    eval_export(
        work_path,
        [sub_param, cal_sub_param, onehot_param, train_param],
        [sub_res, cal_sub_res, onehot_res, train_res],
        storage_config,
        sf_cluster_config,
        expected_input,
        he_mode,
    )

    # by pred comp
    eval_export(
        work_path,
        [sub_param, cal_sub_param, onehot_param, predict_param, score_card_trans_param],
        [sub_res, cal_sub_res, onehot_res, predict_res, score_card_trans_res],
        storage_config,
        sf_cluster_config,
        expected_input,
        he_mode,
    )


@pytest.mark.mpc
def test_model_export_features_in_one_party_true_phe_true(sf_production_setup_comp):
    '''
    In order to accommodate the parallel execution feature of pytest tests to speed up ci pipeline,
    parameters originally configured by parameterize need to be split across two different test_ functions.
    '''
    _inner_test_model_export(
        sf_production_setup_comp, features_in_one_party=True, he_mode=True
    )


@pytest.mark.mpc
def test_model_export_features_in_one_party_false_phe_true(sf_production_setup_comp):
    '''
    In order to accommodate the parallel execution feature of pytest tests to speed up ci pipeline,
    parameters originally configured by parameterize need to be split across two different test_ functions.
    '''
    _inner_test_model_export(
        sf_production_setup_comp, features_in_one_party=False, he_mode=True
    )


@pytest.mark.mpc
def test_model_export_features_in_one_party_false_phe_false(sf_production_setup_comp):
    '''
    In order to accommodate the parallel execution feature of pytest tests to speed up ci pipeline,
    parameters originally configured by parameterize need to be split across two different test_ functions.
    '''
    _inner_test_model_export(
        sf_production_setup_comp, features_in_one_party=False, he_mode=False
    )


@pytest.mark.mpc
def test_model_export_features_in_one_party_true_phe_false(sf_production_setup_comp):
    '''
    In order to accommodate the parallel execution feature of pytest tests to speed up ci pipeline,
    parameters originally configured by parameterize need to be split across two different test_ functions.
    '''
    _inner_test_model_export(
        sf_production_setup_comp, features_in_one_party=True, he_mode=False
    )
