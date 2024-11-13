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

import numpy as np
import pandas as pd
import pytest
from google.protobuf.json_format import MessageToJson, Parse
from pyarrow import orc
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from secretflow.component.core import DistDataType, Storage, build_node_eval_param
from secretflow.component.entry import comp_eval
from secretflow.spec.extend.linear_model_pb2 import GeneralizedLinearModel, LinearModel
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

work_path = f"test_glm"
alice_path = f"{work_path}/x_alice.csv"
bob_path = f"{work_path}/x_bob.csv"
predict_path = f"{work_path}/predict.csv"
predict_referecnce_path = f"{work_path}/predict_reference.csv"


@pytest.fixture
def glm_model(comp_prod_sf_cluster_config):
    model_path = "test_glm/model.sf"
    report_path = "test_glm/model.report"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    storage = Storage(storage_config)

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
            # "input/input_ds/offset":Attribute(ss=[]),
            # "input/input_ds/weight":Attribute(ss=[]),
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

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="1.0.0",
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


@pytest.fixture
def write_complete_data(glm_model, comp_prod_sf_cluster_config):
    pb_path = "test_io/generalized_linear_model_pb"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="1.0.0",
        attr_paths=[
            "generalized_linear_model",
        ],
        attrs=[
            Attribute(b=True),
        ],
        inputs=[glm_model],
        output_uris=[pb_path],
    )
    read_res = comp_eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    generalized_linear_model_pb = GeneralizedLinearModel()
    read_res.outputs[0].meta.Unpack(generalized_linear_model_pb)
    write_data = MessageToJson(generalized_linear_model_pb)
    return write_data


def test_no_change_correct(glm_model, write_data, comp_prod_sf_cluster_config):
    new_glm_model_path = "test_io/new_glm_model"
    pb_path = "test_io/glm_model_pb_unchanged"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    write_param = NodeEvalParam(
        domain="io",
        name="write_data",
        version="1.0.0",
        attr_paths=["write_data", "write_data_type"],
        attrs=[
            Attribute(s=write_data),
            Attribute(s=str(DistDataType.SS_GLM_MODEL)),
        ],
        inputs=[glm_model],
        output_uris=[new_glm_model_path],
    )
    write_res = comp_eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="1.0.0",
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


def test_modify_bias_correct(glm_model, write_data, comp_prod_sf_cluster_config):
    new_rule_path = "test_io/new_bin_rule"
    pb_path = "test_io/rule_pb_unchanged"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config

    read_linear_model_pb = LinearModel()
    Parse(write_data, read_linear_model_pb)
    old_bias = read_linear_model_pb.bias
    read_linear_model_pb.bias += 1
    write_data = MessageToJson(read_linear_model_pb)

    write_param = NodeEvalParam(
        domain="io",
        name="write_data",
        version="1.0.0",
        attr_paths=["write_data", "write_data_type"],
        attrs=[
            Attribute(s=write_data),
            Attribute(s=str(DistDataType.SS_GLM_MODEL)),
        ],
        inputs=[glm_model],
        output_uris=[new_rule_path],
    )
    write_res = comp_eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="1.0.0",
        inputs=[write_res.outputs[0]],
        output_uris=[pb_path],
    )
    read_res = comp_eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    linear_model_pb_changed = LinearModel()
    read_res.outputs[0].meta.Unpack(linear_model_pb_changed)
    np.testing.assert_almost_equal(linear_model_pb_changed.bias, old_bias + 1, 6)


def test_no_change_correct_complete(write_complete_data, comp_prod_sf_cluster_config):
    new_glm_model_path = "test_glm/new_glm_model"
    pb_path = "test_glm/glm_model_pb_unchanged"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    write_param = NodeEvalParam(
        domain="io",
        name="write_data",
        version="1.0.0",
        attr_paths=["write_data", "write_data_type"],
        attrs=[
            Attribute(s=write_complete_data),
            Attribute(s=str(DistDataType.SS_GLM_MODEL)),
        ],
        inputs=[
            DistData(name="null", type=str(DistDataType.NULL)),
        ],
        output_uris=[new_glm_model_path],
    )
    write_res = comp_eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="1.0.0",
        attr_paths=[
            "generalized_linear_model",
        ],
        attrs=[
            Attribute(b=True),
        ],
        inputs=[write_res.outputs[0]],
        output_uris=[pb_path],
    )
    read_res = comp_eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    generalized_linear_model_pb_unchanged = GeneralizedLinearModel()
    read_res.outputs[0].meta.Unpack(generalized_linear_model_pb_unchanged)
    write_data_unchanged = MessageToJson(generalized_linear_model_pb_unchanged)
    # making an exception for hash
    write_data_dict = json.loads(write_complete_data)
    write_data_unchanged_dict = json.loads(write_data_unchanged)
    write_data_unchanged_dict['model']["modelHash"] = write_data_dict.get("model").get(
        "modelHash", ""
    )
    write_data_unchanged = (
        json.dumps(write_data_unchanged_dict).replace("\n", "").replace(" ", "")
    )
    write_complete_data = write_complete_data.replace("\n", "").replace(" ", "")
    # logging.info(f'write_data_unchanged: {write_data_unchanged}')
    # logging.info(f'write_data: {write_complete_data}')
    assert (
        write_data_unchanged == write_complete_data
    ), f"No ops, they should be the same  {write_data_unchanged}, {write_complete_data}"


def get_pred_param(storage_config, sf_cluster_config, model, predict_path):
    self_party = sf_cluster_config.private_config.self_party
    storage = Storage(storage_config)
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

    input_dd = DistData(
        name="train_dataset",
        type="sf.table.vertical_table",
        data_refs=[
            DistData.DataRef(uri=alice_path, party="alice", format="csv"),
            DistData.DataRef(uri=bob_path, party="bob", format="csv"),
        ],
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
    input_dd.meta.Pack(meta)

    predict_param = build_node_eval_param(
        domain="ml.predict",
        name="ss_glm_predict",
        version="1.0.0",
        attrs={
            "receiver": ["alice"],
            "save_ids": False,
            "save_label": True,
            "input/input_ds/saved_features": ["a10", "a2"],
        },
        inputs=[model, input_dd],
        output_uris=[predict_path],
    )
    return predict_param


def predict_reference_data(glm_model, comp_prod_sf_cluster_config):
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    predict_param = get_pred_param(
        storage_config, sf_cluster_config, glm_model, predict_referecnce_path
    )
    predict_res = comp_eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    if "alice" == sf_cluster_config.private_config.self_party:
        storage = Storage(storage_config)
        predict_reference = orc.read_table(
            storage.get_reader(predict_referecnce_path)
        ).to_pandas()
        # logging.warning(f"predict_reference: {predict_reference}")
        return predict_reference


def test_glm_raw_model(glm_model, write_complete_data, comp_prod_sf_cluster_config):
    model = glm_model
    predict_reference = predict_reference_data(model, comp_prod_sf_cluster_config)

    new_glm_path = "test_glm/new_glm"
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    write_param = NodeEvalParam(
        domain="io",
        name="write_data",
        version="1.0.0",
        attr_paths=["write_data", "write_data_type"],
        attrs=[
            Attribute(s=write_complete_data),
            Attribute(s=str(DistDataType.SS_GLM_MODEL)),
        ],
        inputs=[
            DistData(name="null", type=str(DistDataType.NULL)),
        ],
        output_uris=[new_glm_path],
    )
    write_res = comp_eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    predict_param = get_pred_param(
        storage_config, sf_cluster_config, write_res.outputs[0], predict_path
    )
    predict_res = comp_eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
        tracer_report=True,
    )

    logging.info(f"predict tracer_report {predict_res['tracer_report']}")
    assert len(predict_res["eval_result"].outputs) == 1

    self_party = sf_cluster_config.private_config.self_party
    if "alice" == self_party:
        storage = Storage(storage_config)
        input_y = pd.read_csv(storage.get_reader(alice_path))
        output_y = orc.read_table(storage.get_reader(predict_path)).to_pandas()
        # output_y.to_csv(predict_path, index=False)

        # label & pred
        assert output_y.shape[1] == 4
        assert set(output_y.columns) == set(["a2", "a10", "pred", "y"])

        auc = roc_auc_score(input_y["y"], output_y["pred"])
        assert auc > 0.99, f"auc {auc}"

        assert np.allclose(
            output_y["pred"], predict_reference["pred"]
        ), f'output_y["pred"] not close to predict_reference["pred"]'
