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
from secretflow_spec.v1.data_pb2 import DistData

from secretflow.component.core import DistDataType, build_node_eval_param, comp_eval
from tests.component.infra.util import eval_export, get_meta_and_dump_data


def _inner_test_sgb_export(sf_production_setup_comp, features_in_one_party):
    work_path = f"test_sgb_feature_in_one_party{features_in_one_party}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"

    bin_rule_path = f"{work_path}/bin_rule"
    bin_output = f"{work_path}/vert.csv"
    bin_report_path = f"{work_path}/bin_report.json"

    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"

    storage_config, sf_cluster_config = sf_production_setup_comp

    # binning
    feature_selects = [f"a{i}" for i in range(2)] + [f"b{i}" for i in range(2)]
    bin_param = build_node_eval_param(
        domain="preprocessing",
        name="vert_binning",
        version="1.0.0",
        attrs={
            "input/input_ds/feature_selects": feature_selects,
            "bin_num": 4,
        },
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
        output_uris=[bin_output, bin_rule_path, bin_report_path],
    )

    meta = get_meta_and_dump_data(
        work_path,
        sf_production_setup_comp,
        alice_path,
        bob_path,
        features_in_one_party,
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

    # sgb
    train_param = build_node_eval_param(
        domain="ml.train",
        name="sgb_train",
        version="1.0.0",
        attrs={
            "num_boost_round": 2,
            "max_depth": 2,
            "learning_rate": 0.3,
            "objective": "logistic",
            "reg_lambda": 0.1,
            "gamma": 0.5,
            "rowsample_by_tree": 1.0,
            "colsample_by_tree": 1.0,
            "sketch_eps": 0.25,
            "base_score": 0.0,
            "input/input_ds/label": ["y"],
            "input/input_ds/feature_selects": [f"a{i}" for i in range(4)]
            + [f"b{i}" for i in range(4)],
        },
        inputs=[sub_res.outputs[0]],
        output_uris=[model_path],
    )

    train_param.inputs[0].meta.Pack(meta)

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    predict_param = build_node_eval_param(
        domain="ml.predict",
        name="sgb_predict",
        version="1.0.0",
        attrs={"receiver": ["alice"], "save_ids": False, "save_label": True},
        inputs=[train_res.outputs[0], sub_res.outputs[0]],
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
        [sub_param, train_param],
        [sub_res, train_res],
        storage_config,
        sf_cluster_config,
        expected_input,
    )

    # by pred comp
    eval_export(
        work_path,
        [sub_param, predict_param],
        [sub_res, predict_res],
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


@pytest.mark.mpc
def test_sgb_export_features_in_one_party_true(sf_production_setup_comp):
    '''
    In order to accommodate the parallel execution feature of pytest tests to speed up ci pipeline,
    parameters originally configured by parameterize need to be split across two different test_ functions.
    '''
    _inner_test_sgb_export(sf_production_setup_comp, features_in_one_party=True)


@pytest.mark.mpc
def test_sgb_export_features_in_one_party_false(sf_production_setup_comp):
    '''
    In order to accommodate the parallel execution feature of pytest tests to speed up ci pipeline,
    parameters originally configured by parameterize need to be split across two different test_ functions.
    '''
    _inner_test_sgb_export(sf_production_setup_comp, features_in_one_party=False)
