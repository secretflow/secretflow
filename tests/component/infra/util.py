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
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from secretflow.component.data_utils import DistDataType
from secretflow.component.model_export import model_export_comp
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report


import secretflow_serving_lib as sfs
from google.protobuf import json_format


import base64
import copy
import logging
import os
import tarfile


def eval_export(
    dir, comp_params, comp_res, storage_config, sf_cluster_config, expected_input
):
    export_model_path = os.path.join(dir, "s_model.tar.gz")
    report_path = os.path.join(dir, "report")

    input_datasets = []
    output_datasets = []
    component_eval_params = []

    def add_comp(param, res):
        param = copy.deepcopy(param)
        for i in param.inputs:
            input_datasets.append(json_format.MessageToJson(i, indent=0))
        for o in res.outputs:
            output_datasets.append(json_format.MessageToJson(o, indent=0))
        param.ClearField('inputs')
        param.ClearField('output_uris')
        json_param = json_format.MessageToJson(param, indent=0)
        component_eval_params.append(
            base64.b64encode(json_param.encode("utf-8")).decode("utf-8")
        )

    for p, r in zip(comp_params, comp_res):
        add_comp(p, r)

    export_param = NodeEvalParam(
        domain="model",
        name="model_export",
        version="0.0.1",
        attr_paths=[
            "model_name",
            "model_desc",
            "input_datasets",
            "output_datasets",
            "component_eval_params",
        ],
        attrs=[
            Attribute(s="test"),
            Attribute(s="test_desc"),
            Attribute(ss=input_datasets),
            Attribute(ss=output_datasets),
            Attribute(ss=component_eval_params),
        ],
        output_uris=[export_model_path, report_path],
    )

    export_res = model_export_comp.eval(
        param=export_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(export_res.outputs) == 2

    report_dd = export_res.outputs[1]
    report = Report()
    assert report_dd.meta.Unpack(report)
    used_schemas = report.desc.split(",")
    assert set(used_schemas) == set(
        expected_input
    ), f"schemas: {used_schemas}, {expected_input}"

    expected_files = {"model_file", "MANIFEST"}
    comp_storage = ComponentStorage(storage_config)

    if "alice" == sf_cluster_config.private_config.self_party:
        tar_files = dict()
        with tarfile.open(
            fileobj=comp_storage.get_reader(export_model_path),
            mode="r:gz",
        ) as tar:
            for member in tar:
                tar_files[member.name] = tar.extractfile(member.name).read()

        assert expected_files == set(tar_files), f"alice_files {tar_files.keys()}"

        mm = json_format.Parse(tar_files["MANIFEST"], sfs.bundle_pb2.ModelManifest())
        logging.warn(f"alice MANIFEST ............ \n{mm}\n ............ \n")

        mb = sfs.bundle_pb2.ModelBundle()
        mb = json_format.Parse(tar_files["model_file"], sfs.bundle_pb2.ModelBundle())
        logging.warn(f"alice model_file ............ \n{mb}\n ............ \n")

    if "bob" == sf_cluster_config.private_config.self_party:
        tar_files = dict()
        with tarfile.open(
            fileobj=comp_storage.get_reader(export_model_path),
            mode="r:gz",
        ) as tar:
            for member in tar:
                tar_files[member.name] = tar.extractfile(member.name).read()

        assert expected_files == set(tar_files), f"alice_files {tar_files.keys()}"

        mm = json_format.Parse(tar_files["MANIFEST"], sfs.bundle_pb2.ModelManifest())
        logging.warn(f"bob MANIFEST ............ \n{mm}\n ............ \n")

        mb = json_format.Parse(tar_files["model_file"], sfs.bundle_pb2.ModelBundle())
        logging.warn(f"bob model_file ............ \n{mb}\n ............ \n")


def get_ss_sgd_train_param(alice_path, bob_path, model_path, report_path):
    return NodeEvalParam(
        domain="ml.train",
        name="ss_sgd_train",
        version="0.0.1",
        attr_paths=[
            "epochs",
            "learning_rate",
            "batch_size",
            "sig_type",
            "reg_type",
            "penalty",
            "l2_norm",
            "report_weights",
            "input/train_dataset/label",
            "input/train_dataset/feature_selects",
        ],
        attrs=[
            Attribute(i64=1),
            Attribute(f=0.3),
            Attribute(i64=32),
            Attribute(s="t1"),
            Attribute(s="logistic"),
            Attribute(s="l2"),
            Attribute(f=0.05),
            Attribute(b=True),
            Attribute(ss=["y"]),
            Attribute(ss=[f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]),
        ],
        inputs=[
            DistData(
                name="train_dataset",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[model_path, report_path],
    )


def get_eval_param(predict_path):
    return NodeEvalParam(
        domain="ml.eval",
        name="regression_eval",
        version="0.0.1",
        attr_paths=[
            "bucket_size",
            "input/in_ds/label",
            "input/in_ds/prediction",
        ],
        attrs=[
            Attribute(i64=2),
            Attribute(ss=["y"]),
            Attribute(ss=["pred"]),
        ],
        inputs=[
            DistData(
                name="in_ds",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=predict_path, party="alice", format="csv"),
                ],
            ),
        ],
        output_uris=[""],
    )


def get_meta_and_dump_data(
    dir, comp_prod_sf_cluster_config, alice_path, bob_path, features_in_one_party
):
    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)
    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        if features_in_one_party:
            ds = pd.DataFrame(y[:32], columns=["y"])
            ds.to_csv(comp_storage.get_writer(alice_path), index=False)
        else:
            x = pd.DataFrame(x[:32, :15], columns=[f"a{i}" for i in range(15)])
            y = pd.DataFrame(y[:32], columns=["y"])
            ds = pd.concat([x, y], axis=1)
            ds.to_csv(comp_storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        if features_in_one_party:
            ds = pd.DataFrame(
                x[:32, :],
                columns=[f"a{i}" for i in range(15)] + [f"b{i}" for i in range(15)],
            )
            ds.to_csv(comp_storage.get_writer(bob_path), index=False)
        else:
            ds = pd.DataFrame(x[:32, 15:], columns=[f"b{i}" for i in range(15)])
            ds.to_csv(comp_storage.get_writer(bob_path), index=False)

    if features_in_one_party:
        return VerticalTable(
            schemas=[
                TableSchema(
                    feature_types=[],
                    features=[],
                    labels=["y"],
                    label_types=["float32"],
                ),
                TableSchema(
                    feature_types=["float32"] * 30,
                    features=[f"a{i}" for i in range(15)]
                    + [f"b{i}" for i in range(15)],
                ),
            ],
        )
    else:
        return VerticalTable(
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


def get_pred_param(alice_path, bob_path, train_res, predict_path):
    return NodeEvalParam(
        domain="ml.predict",
        name="ss_sgd_predict",
        version="0.0.2",
        attr_paths=[
            "batch_size",
            "receiver",
            "save_ids",
            "save_label",
        ],
        attrs=[
            Attribute(i64=32),
            Attribute(ss=["alice"]),
            Attribute(b=False),
            Attribute(b=True),
        ],
        inputs=[
            train_res.outputs[0],
            DistData(
                name="train_dataset",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[predict_path],
    )
