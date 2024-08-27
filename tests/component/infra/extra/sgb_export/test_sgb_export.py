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
from secretflow.component.ml.boost.sgb.sgb import sgb_predict_comp, sgb_train_comp
from secretflow.component.preprocessing.binning.vert_binning import (
    vert_bin_substitution_comp,
    vert_binning_comp,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from tests.component.infra.util import (
    eval_export,
    get_meta_and_dump_data,
)


@pytest.mark.parametrize("features_in_one_party", [True, False])
def test_sgb_export(comp_prod_sf_cluster_config, features_in_one_party):
    work_path = f"test_sgb_feature_in_one_party{features_in_one_party}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"

    bin_rule_path = f"{work_path}/bin_rule"
    bin_output = f"{work_path}/vert.csv"
    bin_report_path = f"{work_path}/bin_report.json"

    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"

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
        work_path,
        comp_prod_sf_cluster_config,
        alice_path,
        bob_path,
        features_in_one_party,
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
        version="0.0.4",
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
            Attribute(i64=2),
            Attribute(i64=2),
            Attribute(f=0.3),
            Attribute(s="logistic"),
            Attribute(f=0.1),
            Attribute(f=0.5),
            Attribute(f=1),
            Attribute(f=1),
            Attribute(f=0.25),
            Attribute(f=0),
            Attribute(ss=["y"]),
            Attribute(ss=[f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]),
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
        version="0.0.3",
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
