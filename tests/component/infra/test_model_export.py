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


import pytest

from secretflow.component.data_utils import DistDataType
from secretflow.component.entry import comp_eval
from secretflow.component.ml.boost.ss_xgb.ss_xgb import (
    ss_xgb_predict_comp,
    ss_xgb_train_comp,
)
from secretflow.component.ml.linear.ss_sgd import ss_sgd_predict_comp, ss_sgd_train_comp
from secretflow.component.model_export.serving_utils.postprocessing_converter import (
    parse_score_card_transformer_param,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, TableSchema
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from tests.component.infra.util import (
    eval_export,
    get_meta_and_dump_data,
    get_pred_param,
    get_ss_sgd_train_param,
)


@pytest.mark.parametrize("features_in_one_party", [True, False])
def test_ss_sgd_export(comp_prod_sf_cluster_config, features_in_one_party):
    work_path = f"test_ss_sgd_{features_in_one_party}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    report_path = f"{work_path}/model.report"
    predict_path = f"{work_path}/predict.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    train_param = get_ss_sgd_train_param(alice_path, bob_path, model_path, report_path)
    meta = get_meta_and_dump_data(
        work_path,
        comp_prod_sf_cluster_config,
        alice_path,
        bob_path,
        features_in_one_party,
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


@pytest.mark.parametrize("features_in_one_party", [True, False])
def test_ss_xgb_export(comp_prod_sf_cluster_config, features_in_one_party):
    work_path = f"test_xgb_{features_in_one_party}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"

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
            Attribute(i64=2),
            Attribute(i64=2),
            Attribute(f=0.3),
            Attribute(s="logistic"),
            Attribute(f=0.1),
            Attribute(f=1),
            Attribute(f=1),
            Attribute(f=0.25),
            Attribute(f=0),
            Attribute(ss=["y"]),
            Attribute(ss=[f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]),
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
        work_path,
        comp_prod_sf_cluster_config,
        alice_path,
        bob_path,
        features_in_one_party,
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
        version="0.0.2",
        attr_paths=[
            "receiver",
            "save_ids",
            "save_label",
        ],
        attrs=[
            Attribute(ss=["alice"]),
            Attribute(b=False),
            Attribute(b=True),
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


def test_parse_score_card_transformer_param():
    param = NodeEvalParam(
        domain="postprocessing",
        name="score_card_transformer",
        version="1.0.0",
        attr_paths=["input/input_ds/predict_name"],
        attrs=[
            Attribute(ss=["pred"]),
        ],
        inputs=[
            DistData(
                name="input_ds",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri="input.csv", party="alice", format="csv"),
                ],
            )
        ],
        output_uris=["output.csv"],
    )

    meta = IndividualTable(
        schema=TableSchema(
            id_types=None,
            ids=None,
            feature_types=["str", "float"],
            features=["id", "pred"],
        )
    )
    param.inputs[0].meta.Pack(meta)
    res = parse_score_card_transformer_param(param)
    assert res == {
        'positive': True,
        'predict_score_name': 'predict_score',
        'scaled_value': 600,
        'odd_base': 20.0,
        'pdo': 20.0,
        'min_score': 0,
        'max_score': 1000,
    }


def test_score_card_transformer_export(comp_prod_sf_cluster_config):
    work_path = "test_score_card_transformer_export"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"
    score_card_trans_path = f"{work_path}/score_card_trans.csv"

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
            Attribute(ss=[f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]),
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
        work_path,
        comp_prod_sf_cluster_config,
        alice_path,
        bob_path,
        False,
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
        version="0.0.2",
        attr_paths=[
            "receiver",
            "save_ids",
            "save_label",
        ],
        attrs=[
            Attribute(ss=["alice"]),
            Attribute(b=False),
            Attribute(b=True),
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
