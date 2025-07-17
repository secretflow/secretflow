# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import shutil

import numpy as np
import pandas as pd
import pytest
from pyarrow import csv
from secretflow_spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)

from secretflow.component.core import (
    DistDataType,
    VTable,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.component.serving_model_inferencer.serving_model_inferencer import (
    get_output_pred_path,
)


def save_tar_to_buffer(src_tar_path, buffer):
    try:
        with open(src_tar_path, 'rb') as f_src:
            shutil.copyfileobj(f_src, buffer)
        logging.warning(f"{src_tar_path} successfully copy")
    except Exception as e:
        logging.warning(f"copy {src_tar_path} to buffer failed: {e}")


@pytest.mark.parametrize("save_label", [True, False])
@pytest.mark.parametrize("save_features", [True, False])
@pytest.mark.mpc
def test_inferencer(sf_production_setup_comp, save_label, save_features):
    alice_input_path = "serving_model_inferencer/alice.csv"
    bob_input_path = "serving_model_inferencer/bob.orc"
    pred_path = "serving_model_inferencer/pred.csv"
    alice_input_tar_path = "tests/component/infra/serving_model_inferencer/alice/bin_onehot_glm_model.tar.gz"
    bob_input_tar_path = (
        "tests/component/infra/serving_model_inferencer/bob/bin_onehot_glm_model.tar.gz"
    )

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    feature_f1 = -0.5591187559186066
    if self_party == "alice":
        df_alice = pd.DataFrame(
            {
                "id_tttest": [1],
                "f1": [feature_f1],
                "f2": [-0.2814808888968485],
                "f3": [-0.4310124259272412],
                "f4": [0.42698761472064484],
                "f5": [0.6996194412496004],
                "f6": [-0.8971369921102821],
                "f7": [-0.1297223511775294],
                "f8": [-0.6458798434413631],
                "b1": [0.1396383060036957],
                "o1": ["D"],
                "y": [0.0],
            }
        )
        df_alice.to_csv(storage.get_writer(alice_input_path), index=False)
        save_tar_to_buffer(
            alice_input_tar_path, storage.get_writer(alice_input_tar_path)
        )
    elif self_party == "bob":
        df_bob = pd.DataFrame(
            {
                "id_tttest_b": [1],
                "f9": [0.5209858850305862],
                "f10": [0.9053351714559352],
                "f11": [-0.4889431528950445],
                "f12": [0.9829900835655276],
                "f13": [0.1585896989461679],
                "f14": [-0.6509650422128417],
                "f15": [-0.1925996324660748],
                "f16": [0.7337426610542273],
                "b2": [0.2069654682246781],
                "o2": ["C"],
            }
        )
        df_bob.to_orc(
            storage.get_writer(bob_input_path),
            index=False,
        )
        save_tar_to_buffer(bob_input_tar_path, storage.get_writer(bob_input_tar_path))

    receiver = "alice"
    saved_columns = ["id_tttest"]
    if save_features:
        saved_columns.append("f1")
    if save_label:
        saved_columns.append("y")
    param = build_node_eval_param(
        domain="ml.predict",
        name="serving_model_inferencer",
        version="1.0.0",
        attrs={
            "receiver": [receiver],
            "pred_name": "infer_out",
            "input/input_ds/saved_columns": saved_columns,
        },
        inputs=[
            DistData(
                name="input_model",
                type=str(DistDataType.SERVING_MODEL),
                data_refs=[
                    DistData.DataRef(
                        uri=alice_input_tar_path, party="alice", format="tar.gz"
                    ),
                    DistData.DataRef(
                        uri=bob_input_tar_path, party="bob", format="tar.gz"
                    ),
                ],
            ),
            DistData(
                name="input_ds",
                type="sf.table.vertical_table",
                data_refs=[
                    DistData.DataRef(
                        uri=alice_input_path,
                        party="alice",
                        format="csv",
                    ),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="orc"),
                ],
            ),
        ],
        output_uris=[pred_path],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=["int32"],
                ids=["id_tttest"],
                feature_types=[
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "str",
                ],
                features=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "b1", "o1"],
                label_types=["float32"],
                labels=["y"],
            ),
            TableSchema(
                id_types=["int32"],
                ids=["id_tttest_b"],
                feature_types=[
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "str",
                ],
                features=[
                    "f9",
                    "f10",
                    "f11",
                    "f12",
                    "f13",
                    "f14",
                    "f15",
                    "f16",
                    "b2",
                    "o2",
                ],
            ),
        ],
        line_count=1,
    )
    param.inputs[1].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    output_vt = IndividualTable()
    assert res.outputs[0].meta.Unpack(output_vt)
    assert output_vt.line_count == 1, f'output_vt.line_count != 1'

    if self_party == "alice":
        out0 = VTable.from_distdata(res.outputs[0])
        assert len(out0.parties) == 1
        assert "alice" in out0.parties

        alice = out0.parties["alice"]
        expect_columns = ["id_tttest", "infer_out"]
        if save_features:
            expect_columns.extend(["f1"])
        if save_label:
            expect_columns.extend(["y"])
        assert set(alice.columns) == set(expect_columns)

        ds_alice = csv.read_csv(storage.get_reader(alice.uri)).to_pandas()

        logging.warning(f"ds_alice: {ds_alice}")
        predict_reference = [0.393507155146531]
        f1_reference = [feature_f1]
        assert np.allclose(
            ds_alice["infer_out"], predict_reference
        ), f'ds_alice["infer_out"] not close to predict_reference'
        if save_features:
            assert np.allclose(
                ds_alice["f1"], f1_reference
            ), f'ds_alice["f1"] not equal to {feature_f1}'
        if save_label:
            assert np.allclose(ds_alice["y"], 0.0)
        assert np.allclose(ds_alice["id_tttest"], 1)


@pytest.mark.mpc
def test_inferencer_individual(sf_production_setup_comp):
    bob_input_path = "serving_model_inferencer/bob.orc"
    pred_path = "serving_model_inferencer/pred.csv"
    alice_input_tar_path = "tests/component/infra/serving_model_inferencer/alice/features_in_one_party_glm_model.tar.gz"
    bob_input_tar_path = "tests/component/infra/serving_model_inferencer/bob/features_in_one_party_glm_model.tar.gz"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    b1 = 0.416872

    if self_party == "alice":
        save_tar_to_buffer(
            alice_input_tar_path, storage.get_writer(alice_input_tar_path)
        )
    elif self_party == "bob":
        #  bob has all features
        df_bob = pd.DataFrame(
            {
                "id_tttest_b": [1],
                "f1": [-0.156156],
                "f2": [0.409144],
                "f3": [0.752735],
                "f4": [-0.46635],
                "f5": [-0.861575],
                "f6": [-0.571526],
                "f7": [0.937993],
                "f8": [0.19007],
                "b1": [b1],
                "b2": [0.213843],
                "o1": ["C"],
                "o2": ["D"],
                "unused1": [0.60447],
                "unused2": [0.45709],
            }
        )
        df_bob.to_orc(
            storage.get_writer(bob_input_path),
            index=False,
        )
        save_tar_to_buffer(bob_input_tar_path, storage.get_writer(bob_input_tar_path))

    receiver = "bob"
    param = build_node_eval_param(
        domain="ml.predict",
        name="serving_model_inferencer",
        version="1.0.0",
        attrs={
            "receiver": [receiver],
            "pred_name": "infer_out",
            "input/input_ds/saved_columns": ["b1"],
        },
        inputs=[
            DistData(
                name="input_model",
                type=str(DistDataType.SERVING_MODEL),
                data_refs=[
                    DistData.DataRef(
                        uri=alice_input_tar_path, party="alice", format="tar.gz"
                    ),
                    DistData.DataRef(
                        uri=bob_input_tar_path, party="bob", format="tar.gz"
                    ),
                ],
            ),
            DistData(
                name="input_ds",
                type="sf.table.individual",
                data_refs=[
                    DistData.DataRef(
                        uri=bob_input_path,
                        party="bob",
                        format="orc",
                    ),
                ],
            ),
        ],
        output_uris=[pred_path],
    )

    meta = IndividualTable(
        schema=TableSchema(
            id_types=["int32"],
            ids=["id_tttest_b"],
            feature_types=["float32"] * 12 + ["str"] * 2,
            features=[f"f{i + 1}" for i in range(8)]
            + ["unused1", "unused2", "b1", "b2", "o1", "o2"],
        ),
    )
    param.inputs[1].meta.Pack(meta)

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    if self_party == "bob":
        out0 = VTable.from_distdata(res.outputs[0])
        assert len(out0.parties) == 1
        assert "bob" in out0.parties

        bob = out0.parties["bob"]

        expect_columns = ["id_tttest_b", "b1", "infer_out"]
        assert set(bob.columns) == set(expect_columns)
        ds_bob = csv.read_csv(storage.get_reader(bob.uri)).to_pandas()

        logging.warning(f"ds_bob: {ds_bob}")
        predict_reference = [0.459614]
        b1_reference = [b1]
        assert np.allclose(
            ds_bob["infer_out"], predict_reference
        ), f'ds_bob["infer_out"] not close to predict_reference'
        assert np.allclose(
            ds_bob["b1"], b1_reference
        ), f'ds_bob["b1"] not equal to {b1}'

        assert np.allclose(ds_bob["id_tttest_b"], 1)


def test_is_safe_path():
    base_path = "/home/user/bob"

    pred_path = "pred.csv"
    assert get_output_pred_path(base_path, pred_path) == "/home/user/bob/pred.csv"

    pred_path = "../pred.csv"
    with pytest.raises(
        AssertionError,
        match=f'path: {pred_path} contains .., which is unsafe',
    ):
        get_output_pred_path(base_path, pred_path)
