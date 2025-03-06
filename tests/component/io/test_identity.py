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

import json
import logging

import pandas as pd
import pytest
from google.protobuf.json_format import MessageToJson
from secretflow_spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from secretflow.component.core import build_node_eval_param, comp_eval, make_storage
from secretflow.spec.extend.linear_model_pb2 import LinearModel


@pytest.fixture
def glm_model(comp_prod_sf_cluster_config):
    alice_path = "test_glm/x_alice.csv"
    bob_path = "test_glm/x_bob.csv"
    model_path = "test_glm/model.sf"
    report_path = "test_glm/model.report"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
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
        name="ss_glm_train",
        version="1.0.0",
        attrs={
            "epochs": 3,
            "learning_rate": 0.25,
            "batch_size": 128,
            "link_type": "Logit",
            "label_dist_type": "Bernoulli",
            "optimizer": "SGD",
            "l2_lambda": 0.3,
            "report_weights": True,
            "input/input_ds/label": ["y"],
            "input/input_ds/feature_selects": [f"a{i}" for i in range(15)]
            + [f"b{i}" for i in range(15)],
            # "input/input_ds/offset": Attribute(ss=[]),
            # "input/input_ds/weight": Attribute(ss=[]),
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
        output_uris=[model_path, report_path],
    )

    meta = VerticalTable(
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
    train_param.inputs[0].meta.Pack(meta)

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    return train_res.outputs[0]


@pytest.fixture
def write_data(glm_model, comp_prod_sf_cluster_config):
    pb_path = "test_io/linear_model_pb"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    read_param = build_node_eval_param(
        domain="io",
        name="read_data",
        version="1.0.0",
        attrs=None,
        inputs=[glm_model],
        output_uris=[pb_path],
    )
    read_res = comp_eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    linear_model_pb = LinearModel()
    read_res.outputs[0].meta.Unpack(linear_model_pb)
    write_data = MessageToJson(linear_model_pb)
    return write_data


def test_glm_model_correct(glm_model, write_data, comp_prod_sf_cluster_config):
    new_glm_model_path = "test_io/new_glm_model"
    pb_path = "test_io/glm_model_pb_unchanged"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    identity_param = build_node_eval_param(
        domain="io",
        name="identity",
        version="1.0.0",
        attrs=None,
        inputs=[glm_model],
        output_uris=[new_glm_model_path],
    )
    write_res = comp_eval(
        param=identity_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    read_param = build_node_eval_param(
        domain="io",
        name="read_data",
        version="1.0.0",
        attrs=None,
        inputs=[write_res.outputs[0]],
        output_uris=[pb_path],
    )
    read_res = comp_eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    linear_model_pb_unchanged = LinearModel()
    read_res.outputs[0].meta.Unpack(linear_model_pb_unchanged)
    write_data_unchanged = MessageToJson(linear_model_pb_unchanged)
    # making an exception for hash
    write_data_dict = json.loads(write_data)
    write_data_unchanged_dict = json.loads(write_data_unchanged)
    write_data_unchanged_dict["modelHash"] = write_data_dict.get("modelHash", "")
    write_data_unchanged = (
        json.dumps(write_data_unchanged_dict).replace("\n", "").replace(" ", "")
    )
    write_data = write_data.replace("\n", "").replace(" ", "")
    logging.info(write_data_unchanged)
    assert (
        write_data_unchanged == write_data
    ), f"No ops, they should be the same  {write_data_unchanged}, {write_data}"
