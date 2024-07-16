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

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from secretflow.component.ml.linear.ss_glm import ss_glm_predict_comp, ss_glm_train_comp
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report


@pytest.mark.parametrize("optimizer", ["SGD", "IRLS"])
@pytest.mark.parametrize("with_checkpoint", [True, False])
def test_glm(comp_prod_sf_cluster_config, optimizer, with_checkpoint):
    work_path = f"test_glm_{optimizer}_{with_checkpoint}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    report_path = f"{work_path}/model.report"
    checkpoint_path = f"{work_path}/checkpoint"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        x["id1"] = pd.Series([f"{i}" for i in range(x.shape[0])])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(comp_storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds["id2"] = pd.Series([f"{i}" for i in range(x.shape[0])])
        ds.to_csv(comp_storage.get_writer(bob_path), index=False)

    train_param = NodeEvalParam(
        domain="ml.train",
        name="ss_glm_train",
        version="0.0.3",
        attr_paths=[
            "epochs",
            "learning_rate",
            "batch_size",
            "link_type",
            "label_dist_type",
            "optimizer",
            "l2_lambda",
            "infeed_batch_size_limit",
            "iter_start_irls",
            "stopping_rounds",
            "stopping_tolerance",
            "stopping_metric",
            "report_weights",
            "input/train_dataset/label",
            "input/train_dataset/feature_selects",
            "input/train_dataset/offset",
            "input/train_dataset/weight",
            "report_metric",
        ],
        attrs=[
            Attribute(i64=10),
            Attribute(f=0.3),
            Attribute(i64=128),
            Attribute(s="Logit"),
            Attribute(s="Bernoulli"),
            Attribute(s=optimizer),
            Attribute(f=0.3),
            Attribute(i64=128 * 30),
            Attribute(i64=1),
            Attribute(i64=2),
            Attribute(f=0.01),
            Attribute(s="RMSE"),
            Attribute(b=True),
            Attribute(ss=["y"]),
            Attribute(ss=[f"a{i}" for i in range(15)] + [f"b{i}" for i in range(15)]),
            Attribute(ss=[]),
            Attribute(ss=[]),
            Attribute(b=True),
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
        output_uris=[model_path, report_path],
        checkpoint_uri=checkpoint_path if with_checkpoint else "",
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                ids=["id1"],
                id_types=["str"],
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
                labels=["y"],
                label_types=["float32"],
            ),
            TableSchema(
                ids=["id2"],
                id_types=["str"],
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    train_param.inputs[0].meta.Pack(meta)

    train_res = ss_glm_train_comp.eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
        tracer_report=True,
    )

    logging.info(f"train tracer_report {train_res['tracer_report']}")
    comp_ret = Report()
    train_res = train_res["eval_result"]
    train_res.outputs[1].meta.Unpack(comp_ret)
    logging.info(comp_ret)

    def run_pred(predict_path, train_res):
        predict_param = NodeEvalParam(
            domain="ml.predict",
            name="ss_glm_predict",
            version="0.0.2",
            attr_paths=[
                "receiver",
                "save_ids",
                "save_label",
                "input/feature_dataset/saved_features",
            ],
            attrs=[
                Attribute(ss=["alice"]),
                Attribute(b=True),
                Attribute(b=True),
                Attribute(ss=["a10", "a2"]),
            ],
            inputs=[train_res.outputs[0], train_param.inputs[0]],
            output_uris=[predict_path],
        )

        predict_res = ss_glm_predict_comp.eval(
            param=predict_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
            tracer_report=True,
        )

        logging.info(f"predict tracer_report {predict_res['tracer_report']}")
        assert len(predict_res["eval_result"].outputs) == 1

        if "alice" == sf_cluster_config.private_config.self_party:
            comp_storage = ComponentStorage(storage_config)
            input_y = pd.read_csv(comp_storage.get_reader(alice_path))
            dtype = defaultdict(np.float32)
            dtype["id1"] = np.string_
            output_y = pd.read_csv(comp_storage.get_reader(predict_path), dtype=dtype)

            # label & pred
            assert output_y.shape[1] == 5

            assert set(output_y.columns) == set(["a2", "a10", "pred", "y", "id1"])

            if self_party == "alice":
                for n in ["a2", "a10", "y"]:
                    assert np.allclose(ds[n].values, output_y[n].values)
                assert np.all(ds["id1"].values == output_y["id1"].values)

            assert input_y.shape[0] == output_y.shape[0]

            auc = roc_auc_score(input_y["y"], output_y["pred"])
            assert auc > 0.99, f"auc {auc}"

    run_pred(f"{work_path}/predict.csv", train_res)

    if with_checkpoint:
        cp_num = len(comp_ret.tabs[1].divs[0].children[0].table.rows)
        if "alice" == sf_cluster_config.private_config.self_party:
            comp_storage = ComponentStorage(storage_config)
            for i in range(int(cp_num / 2), cp_num):
                with comp_storage.get_writer(f"{checkpoint_path}_{i}") as f:
                    # destroy some checkpoint to rollback train progress
                    f.write(b"....")

        # run train again from checkpoint
        train_param.output_uris[0] = f"{work_path}/model.sf.2"
        train_res = ss_glm_train_comp.eval(
            param=train_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        run_pred(f"{work_path}/predict.csv.2", train_res)
