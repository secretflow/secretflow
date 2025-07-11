# Copyright 2022 Ant Group Co., Ltd.
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

import os
import time

import pytest
from sklearn.metrics import mean_squared_error, roc_auc_score

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ml.boost.ss_xgb_v import Xgb
from secretflow.utils.simulation.datasets import load_dermatology, load_linear
from tests.sf_fixtures import SFProdParams

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _run_xgb(env, test_name, v_data, label_data, y, logistic, subsample, colsample):
    xgb = Xgb(env.spu)
    xgb02 = Xgb([env.spu, env.spu])
    start = time.time()
    params = {
        'num_boost_round': 2,
        'max_depth': 3,
        'sketch_eps': 0.25,
        'objective': 'logistic' if logistic else 'linear',
        'reg_lambda': 0.1,
        'subsample': subsample,
        'colsample_by_tree': colsample,
        'base_score': 0.5,
    }
    model = xgb.train(params.copy(), v_data, label_data)
    model02 = xgb02.train(params.copy(), v_data, label_data)
    wait(model.weights[-1])
    print(f"{test_name} train time: {time.time() - start}")
    start = time.time()
    spu_yhat = model.predict(v_data)
    yhat = reveal(spu_yhat)
    spu_yhat_02 = model02.predict(v_data)
    yhat02 = reveal(spu_yhat_02)
    print(f"{test_name} predict time: {time.time() - start}")
    if logistic:
        print(f"{test_name} auc: {roc_auc_score(y, yhat)}")
        print(f"{test_name} auc02: {roc_auc_score(y, yhat02)}")
    else:
        print(f"{test_name} mse: {mean_squared_error(y, yhat)}")
        print(f"{test_name} mse02: {mean_squared_error(y, yhat02)}")

    fed_yhat = model.predict(v_data, env.alice)
    assert len(fed_yhat.partitions) == 1 and env.alice in fed_yhat.partitions
    yhat = reveal(fed_yhat.partitions[env.alice])
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    if logistic:
        print(f"{test_name} auc: {roc_auc_score(y, yhat)}")
    else:
        print(f"{test_name} mse: {mean_squared_error(y, yhat)}")


def _run_npc_linear(env, test_name, parts, label_device):
    vdf = load_linear(parts=parts)

    label_data = vdf['y']
    y = reveal(label_data.partitions[label_device].data).values
    label_data = (label_data.values)[:500, :]
    y = y[:500, :]

    v_data = vdf.drop(columns="y").values
    v_data = v_data[:500, :]
    label_data = label_data[:500, :]

    _run_xgb(env, test_name, v_data, label_data, y, True, 0.9, 1)


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_2pc_linear(sf_production_setup_devices):
    devices = sf_production_setup_devices

    parts = {
        devices.alice: (1, 11),
        devices.bob: (11, 22),
    }
    _run_npc_linear(devices, "2pc_linear", parts, devices.bob)


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_3pc_linear(sf_production_setup_devices):
    devices = sf_production_setup_devices

    parts = {
        devices.alice: (1, 8),
        devices.bob: (8, 16),
        devices.carol: (16, 22),
    }
    _run_npc_linear(devices, "3pc_linear", parts, devices.carol)


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_breast_cancer(sf_production_setup_devices):
    from sklearn.datasets import load_breast_cancer

    devices = sf_production_setup_devices

    ds = load_breast_cancer()
    x, y = ds['data'], ds['target']

    v_data = FedNdarray(
        {
            devices.alice: (devices.alice(lambda: x[:, :15])()),
            devices.bob: (devices.bob(lambda: x[:, 15:])()),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label_data = FedNdarray(
        {devices.alice: (devices.alice(lambda: y)())},
        partition_way=PartitionWay.VERTICAL,
    )

    _run_xgb(
        devices,
        "breast_cancer",
        v_data,
        label_data,
        y,
        True,
        1,
        0.9,
    )


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_dermatology(sf_production_setup_devices):
    devices = sf_production_setup_devices

    vdf = load_dermatology(
        parts={
            devices.alice: (0, 17),
            devices.bob: (17, 35),
        },
        axis=1,
    ).fillna(0)

    label_data = vdf['class']
    y = reveal(label_data.partitions[devices.bob].data).values
    v_data = vdf.drop(columns="class").values
    label_data = label_data.values

    _run_xgb(
        devices,
        "dermatology",
        v_data,
        label_data,
        y,
        False,
        0.9,
        0.9,
    )


# TODO(fengjun.feng): move the following to a seperate integration test.

# if __name__ == '__main__':
#     # HOW TO RUN:
#     # 0. change args following <<< !!! >>> flag.
#     #    you need change input data path & train settings before run.
#     # 1. install requirements following INSTALLATION.md
#     # 2. set env
#     #    export PYTHONPATH=$PYTHONPATH:bazel-bin
#     # 3. run
#     #    python tests/ml/boost/ss_xgb_v/test_vert_ss_xgb.py

#     # use aby3 in this example.
#     cluster = ABY3MultiDriverDeviceTestCase()
#     cluster.setUpClass()
#     # init log
#     logging.getLogger().setLevel(logging.INFO)

#     # prepare data
#     start = time.time()
#     # read dataset.
#     vdf = create_df(
#         # load file 'dataset('linear')' as train dataset.
#         # <<< !!! >>> replace dataset path to your own local file.
#         dataset('linear'),
#         # split 1-10 columns to alice and 11-22 columns to bob which include y col.
#         # <<< !!! >>> replace parts range to your own dataset's columns count.
#         parts={cluster.alice: (1, 11), cluster.bob: (11, 22)},
#         # split by vertical. DON'T change this.
#         axis=1,
#     )
#     # split y out of dataset,
#     # <<< !!! >>> change 'y' if label column name is not y in dataset.
#     label_data = vdf['y']
#     # v_data remains all features.
#     v_data = vdf.drop(columns="y")
#     # <<< !!! >>> change cluster.bob if y not belong to bob.
#     y = reveal(label_data.partitions[cluster.bob].data)
#     wait([p.data for p in v_data.partitions.values()])
#     logging.info(f"IO times: {time.time() - start}s")

#     xgb = Xgb(cluster.spu)

#     params = {
#         # <<< !!! >>> change args to your test settings.
#         # for more detail, see Xgb.train.__doc__
#         'num_boost_round': 3,
#         'max_depth': 3,
#         'learning_rate': 0.3,
#         'sketch_eps': 0.25,
#         'objective': 'logistic',
#         'reg_lambda': 0.1,
#         'subsample': 1,
#         'colsample_by_tree': 1,
#         'base_score': 0.5,
#     }
#     start = time.time()
#     model = xgb.train(params, v_data, label_data)
#     logging.info(f"main train time: {time.time() - start}")
#     start = time.time()
#     spu_yhat = model.predict(v_data)
#     yhat = reveal(spu_yhat)
#     logging.info(f"main predict time: {time.time() - start}")
#     logging.info(f"main auc: {roc_auc_score(y, yhat)}")
