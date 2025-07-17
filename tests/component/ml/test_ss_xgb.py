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

import pandas as pd
import pytest
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from secretflow.component.core import (
    VTable,
    VTableParty,
    build_node_eval_param,
    comp_eval,
    make_storage,
)

NUM_BOOST_ROUND = 3


@pytest.mark.parametrize("with_checkpoint", [True, False])
@pytest.mark.mpc
def test_ss_xgb(sf_production_setup_comp, with_checkpoint):
    work_path = f"ut_test_ss_xgb_{with_checkpoint}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"
    checkpoint_path = f"{work_path}/checkpoint"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(storage.get_writer(bob_path), index=False)

    train_param = build_node_eval_param(
        domain="ml.train",
        name="ss_xgb_train",
        version="1.0.0",
        attrs={
            "num_boost_round": NUM_BOOST_ROUND,
            "max_depth": 3,
            "learning_rate": 0.3,
            "objective": "logistic",
            "reg_lambda": 0.1,
            "subsample": 1.0,
            "colsample_by_tree": 1.0,
            "sketch_eps": 0.25,
            "base_score": 0.0,
            "input/input_ds/label": ["y"],
            "input/input_ds/feature_selects": [f"a{i}" for i in range(15)]
            + [f"b{i}" for i in range(15)],
        },
        inputs=[
            VTable(
                name="train_dataset",
                parties=[
                    VTableParty.from_dict(
                        uri=alice_path,
                        party="alice",
                        format="csv",
                        features={f"a{i}": "float32" for i in range(15)},
                        labels={"y": "float32"},
                    ),
                    VTableParty.from_dict(uri=bob_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[model_path],
        checkpoint_uri=checkpoint_path if with_checkpoint else "",
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

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    def run_pred(predict_path, train_res):
        predict_param = build_node_eval_param(
            domain="ml.predict",
            name="ss_xgb_predict",
            version="1.0.0",
            attrs={
                "receiver": ["alice"],
                "save_ids": False,
                "save_label": True,
                "saved_features": ["a2", "a10"],
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

        predict_res = comp_eval(
            param=predict_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        assert len(predict_res.outputs) == 1

        if "alice" == sf_cluster_config.private_config.self_party:
            storage = make_storage(storage_config)
            input_y = pd.read_csv(storage.get_reader(alice_path))
            output_y = orc.read_table(storage.get_reader(predict_path)).to_pandas()

            # label & pred
            assert output_y.shape[1] == 4

            assert input_y.shape[0] == output_y.shape[0]

            auc = roc_auc_score(input_y["y"], output_y["pred"])
            assert auc > 0.99, f"auc {auc}"

    run_pred(predict_path, train_res)
    if with_checkpoint:
        cp_num = NUM_BOOST_ROUND
        if "alice" == sf_cluster_config.private_config.self_party:
            storage = make_storage(storage_config)
            for i in range(int(cp_num / 2), cp_num):
                with storage.get_writer(f"{checkpoint_path}_{i}") as f:
                    # destroy some checkpoint to rollback train progress
                    f.write(b"....")

        # run train again from checkpoint
        train_param.output_uris[0] = f"{work_path}/model.sf.2"
        train_res = comp_eval(
            param=train_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        run_pred(f"{work_path}/predict.csv.2", train_res)
