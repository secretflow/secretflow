# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest
from google.protobuf.json_format import MessageToJson
from pyarrow import orc
from secretflow_spec.v1.data_pb2 import DistData

from secretflow.component.core import (
    DistDataType,
    build_node_eval_param,
    comp_eval,
    make_storage,
)
from secretflow.spec.extend.sgb_model_pb2 import SgbModel
from tests.component.ml.test_sgb import (
    get_meta_and_dump_data,
    get_pred_param,
    get_train_param,
)

work_path = f"test_io_sgb"
alice_path = f"{work_path}/x_alice.csv"
bob_path = f"{work_path}/x_bob.csv"
predict_path = f"{work_path}/predict.csv"


def gen_sgb_model_train_res(sf_production_setup_comp):
    work_path = f"test_io_sgb"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    report_path = f"{work_path}/report.sf"

    storage_config, sf_cluster_config = sf_production_setup_comp

    train_param = get_train_param(
        alice_path,
        bob_path,
        report_path,
        model_path,
        "",
        "1.1.0",
    )
    meta = get_meta_and_dump_data(sf_production_setup_comp, alice_path, bob_path)
    train_param.inputs[0].meta.Pack(meta)

    train_res = comp_eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    return train_res


@pytest.mark.mpc
def sgb_model_write_data(sgb_model, sf_production_setup_comp):
    pb_path = "test_io/sgb_model_pb"
    storage_config, sf_cluster_config = sf_production_setup_comp

    read_param = build_node_eval_param(
        domain="io",
        name="read_data",
        version="1.0.0",
        attrs=None,
        inputs=[sgb_model],
        output_uris=[pb_path],
    )
    read_res = comp_eval(
        param=read_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    sgb_model_pb = SgbModel()
    read_res.outputs[0].meta.Unpack(sgb_model_pb)
    write_data = MessageToJson(sgb_model_pb)
    return write_data


def predict_reference_data(train_res, sf_production_setup_comp):
    storage_config, sf_cluster_config = sf_production_setup_comp
    predict_param = get_pred_param(alice_path, bob_path, train_res, predict_path)
    meta = get_meta_and_dump_data(sf_production_setup_comp, alice_path, bob_path)
    predict_param.inputs[1].meta.Pack(meta)
    predict_res = comp_eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    if "alice" == sf_cluster_config.private_config.self_party:
        storage = make_storage(storage_config)
        output_y = orc.read_table(storage.get_reader(predict_path)).to_pandas()
        return output_y


@pytest.mark.mpc
def test_no_change_correct(sf_production_setup_comp):
    new_sgb_model_path = "test_io/new_sgb_model"
    pb_path = "test_io/sgb_model_pb_unchanged"
    train_res = gen_sgb_model_train_res(sf_production_setup_comp)
    write_data = sgb_model_write_data(train_res.outputs[0], sf_production_setup_comp)
    storage_config, sf_cluster_config = sf_production_setup_comp
    write_param = build_node_eval_param(
        domain="io",
        name="write_data",
        version="1.0.0",
        attrs={
            "write_data": write_data,
            "write_data_type": str(DistDataType.SGB_MODEL),
        },
        inputs=[train_res.outputs[0]],
        output_uris=[new_sgb_model_path],
    )
    write_res = comp_eval(
        param=write_param,
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
    sgb_model_pb_unchanged = SgbModel()
    read_res.outputs[0].meta.Unpack(sgb_model_pb_unchanged)
    write_data_unchanged = MessageToJson(sgb_model_pb_unchanged)

    write_data_unchanged_dict = json.loads(write_data_unchanged)
    write_data_unchanged = (
        json.dumps(write_data_unchanged_dict).replace("\n", "").replace(" ", "")
    )
    write_data = write_data.replace("\n", "").replace(" ", "")
    # logging.warning(f'write_data: {write_data}')
    # logging.info(write_data_unchanged)
    assert (
        write_data_unchanged == write_data
    ), f"No ops, they should be the same  {write_data_unchanged}, {write_data}"


@pytest.mark.mpc
def test_write_predict_data(sf_production_setup_comp):
    train_res = gen_sgb_model_train_res(sf_production_setup_comp)
    predict_reference = predict_reference_data(train_res, sf_production_setup_comp)
    write_data = sgb_model_write_data(train_res.outputs[0], sf_production_setup_comp)
    new_sgb_model_path = "test_io/new_sgb_model"
    storage_config, sf_cluster_config = sf_production_setup_comp
    write_param = build_node_eval_param(
        domain="io",
        name="write_data",
        version="1.0.0",
        attrs={
            "write_data": write_data,
            "write_data_type": str(DistDataType.SGB_MODEL),
        },
        inputs=[
            DistData(name="null", type=str(DistDataType.NULL)),
        ],
        output_uris=[new_sgb_model_path],
    )
    write_res = comp_eval(
        param=write_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    assert len(write_res.outputs) == 1
    predict_param = get_pred_param(alice_path, bob_path, write_res, predict_path)
    meta = get_meta_and_dump_data(sf_production_setup_comp, alice_path, bob_path)
    predict_param.inputs[1].meta.Pack(meta)

    storage_config, sf_cluster_config = sf_production_setup_comp
    predict_res = comp_eval(
        param=predict_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(predict_res.outputs) == 1

    if "alice" == sf_cluster_config.private_config.self_party:
        storage = make_storage(storage_config)
        output_y = orc.read_table(storage.get_reader(predict_path)).to_pandas()

        assert output_y["pred"].equals(
            predict_reference["pred"]
        ), 'output_y["pred"] != predict_reference["pred"]'
