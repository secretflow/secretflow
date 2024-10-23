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


import time

import pytest

from secretflow.component.core import build_node_eval_param
from secretflow.component.entry import comp_eval
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
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
def test_ss_sgd_export(comp_prod_sf_cluster_config, features_in_one_party, he_mode):
    work_path = f"test_ss_sgd_{features_in_one_party}_{he_mode}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    report_path = f"{work_path}/model.report"
    predict_path = f"{work_path}/predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    sf_cluster_config = setup_cluster_config(sf_cluster_config, he_mode)

    train_param = get_ss_sgd_train_param(alice_path, bob_path, model_path, report_path)
    meta = get_meta_and_dump_data(
        work_path,
        comp_prod_sf_cluster_config,
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
def test_ss_xgb_export(comp_prod_sf_cluster_config, features_in_one_party):
    work_path = f"test_xgb_{features_in_one_party}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

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
        comp_prod_sf_cluster_config,
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
    )

    # by pred comp
    eval_export(
        work_path,
        [predict_param],
        [predict_res],
        storage_config,
        sf_cluster_config,
        expected_input,
    )

    with pytest.raises(
        AssertionError, match="feature not supported yet. change `he_mode` to False."
    ):
        # by train comp
        eval_export(
            work_path,
            [train_param],
            [train_res],
            storage_config,
            sf_cluster_config,
            expected_input,
            True,
        )
    time.sleep(4)

    with pytest.raises(
        AssertionError, match="feature not supported yet. change `he_mode` to False."
    ):
        # by pred comp
        eval_export(
            work_path,
            [predict_param],
            [predict_res],
            storage_config,
            sf_cluster_config,
            expected_input,
            True,
        )
    time.sleep(4)


def test_score_card_transformer_export(comp_prod_sf_cluster_config):
    work_path = "test_score_card_transformer_export"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"
    score_card_trans_path = f"{work_path}/score_card_trans.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

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
        comp_prod_sf_cluster_config,
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
    score_card_trans_param = NodeEvalParam(
        domain="postprocessing",
        name="score_card_transformer",
        version="1.0.0",
        attr_paths=[
            "positive",
            "predict_score_name",
            "scaled_value",
            "odd_base",
            "pdo",
            "input/input_ds/predict_name",
        ],
        attrs=[
            Attribute(i64=1),
            Attribute(s="predict_score"),
            Attribute(i64=600),
            Attribute(f=20),
            Attribute(f=20),
            Attribute(ss=["pred"]),
        ],
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

    with pytest.raises(
        AssertionError, match="feature not supported yet. change `he_mode` to False."
    ):
        # by train comp
        eval_export(
            work_path,
            [train_param],
            [train_res],
            storage_config,
            sf_cluster_config,
            expected_input,
            True,
        )
    time.sleep(4)

    with pytest.raises(
        AssertionError, match="feature not supported yet. change `he_mode` to False."
    ):
        # by pred comp
        eval_export(
            work_path,
            [predict_param],
            [predict_res],
            storage_config,
            sf_cluster_config,
            expected_input,
            True,
        )
    time.sleep(4)
