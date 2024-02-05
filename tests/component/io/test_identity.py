import json
import logging
import os

import pandas as pd
import pytest
from google.protobuf.json_format import MessageToJson
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from secretflow.component.io.identity import identity
from secretflow.component.io.io import io_read_data
from secretflow.component.ml.linear.ss_glm import ss_glm_train_comp
from secretflow.spec.extend.linear_model_pb2 import LinearModel
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


@pytest.fixture
def glm_model(comp_prod_sf_cluster_config):
    alice_path = "test_glm/x_alice.csv"
    bob_path = "test_glm/x_bob.csv"
    model_path = "test_glm/model.sf"
    report_path = "test_glm/model.report"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    local_fs_wd = storage_config.local_fs.wd

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        os.makedirs(
            os.path.join(local_fs_wd, "test_glm"),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(os.path.join(local_fs_wd, alice_path), index=False)

    elif self_party == "bob":
        os.makedirs(
            os.path.join(local_fs_wd, "test_glm"),
            exist_ok=True,
        )

        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(os.path.join(local_fs_wd, bob_path), index=False)

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
            Attribute(i64=3),
            Attribute(f=0.3),
            Attribute(i64=128),
            Attribute(s="Logit"),
            Attribute(s="Bernoulli"),
            Attribute(s="SGD"),
            Attribute(f=0.3),
            Attribute(b=True),
            Attribute(ss=["y"]),
            Attribute(ss=[f"a{i}" for i in range(15)] + [f"b{i}" for i in range(15)]),
            Attribute(ss=[]),
            Attribute(ss=[]),
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

    train_res = ss_glm_train_comp.eval(
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
        version="0.0.1",
        inputs=[glm_model],
        output_uris=[pb_path],
    )
    read_res = io_read_data.eval(
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
    identity_param = NodeEvalParam(
        domain="io",
        name="identity",
        version="0.0.1",
        inputs=[glm_model],
        output_uris=[new_glm_model_path],
    )
    write_res = identity.eval(
        param=identity_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    read_param = NodeEvalParam(
        domain="io",
        name="read_data",
        version="0.0.1",
        inputs=[write_res.outputs[0]],
        output_uris=[pb_path],
    )
    read_res = io_read_data.eval(
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
