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

import copy
import logging
import time

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ml.linear.ss_glm import SSGLM
from secretflow.ml.linear.ss_glm.core import get_dist
from tests.sf_fixtures import SFProdParams


def _transform(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def _wait_io(inputs):
    wait_objs = list()
    for input in inputs:
        wait_objs.extend([input.partitions[d] for d in input.partitions])
    wait(wait_objs)


def _run_test(
    devices, test_name, v_data, label_data, y, batch_size, link, dist, l2_lambda=None
):
    model = SSGLM(devices.spu)
    label_data_copy = copy.deepcopy(label_data)

    start = time.time()
    model.fit_sgd(
        v_data,
        label_data,
        None,
        None,
        3,
        link,
        dist,
        1,
        1,
        0.3,
        iter_start_irls=1,
        batch_size=batch_size,
        l2_lambda=l2_lambda,
        stopping_rounds=0,
    )
    logging.info(f"{test_name} sgb train time: {time.time() - start}")
    start = time.time()

    spu_yhat = model.predict(v_data)
    yhat = reveal(spu_yhat)
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    logging.info(f"{test_name} predict time: {time.time() - start}")
    deviance = get_dist(dist, 1, 1.65).deviance(yhat, y.reshape(-1, 1), None)
    # deviance = get_dist(dist, 1, 1).deviance(yhat, y, None)
    logging.info(f"{test_name} deviance: {deviance}")

    start = time.time()
    model.fit_irls(
        v_data,
        label_data_copy,
        None,
        None,
        10,
        link,
        dist,
        1,
        1,
        l2_lambda=l2_lambda,
        stopping_rounds=1,
        stopping_tolerance=0.001,
        report_metric=False,
    )
    logging.info(f"{test_name} irls train time: {time.time() - start}")
    start = time.time()
    spu_yhat = model.predict(v_data)
    yhat = reveal(spu_yhat)
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    logging.info(f"{test_name} predict time: {time.time() - start}")
    deviance = get_dist(dist, 1, 1.65).deviance(yhat, y.reshape(-1, 1), None)
    logging.info(f"{test_name} deviance: {deviance}")

    fed_yhat = model.predict(v_data, to_pyu=devices.alice)
    assert len(fed_yhat.partitions) == 1 and devices.alice in fed_yhat.partitions
    yhat = reveal(fed_yhat.partitions[devices.alice])
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    deviance = get_dist(dist, 1, 1.65).deviance(yhat, y.reshape(-1, 1), None)
    logging.info(f"{test_name} deviance: {deviance}")

    fed_w, bias = model.spu_w_to_federated(v_data, devices.alice)
    wait([*list(fed_w.partitions.values()), bias])
    yhat = reveal(model.predict_fed_w(v_data, fed_w, bias))
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    deviance = get_dist(dist, 1, 1).deviance(yhat, y, None)
    logging.info(f"{test_name} fed deviance: {deviance}")


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_breast_cancer(sf_production_setup_devices):
    from sklearn.datasets import load_breast_cancer

    devices = sf_production_setup_devices
    start = time.time()

    ds = load_breast_cancer()
    x, y = _transform(ds['data']), ds['target']

    v_data = FedNdarray(
        partitions={
            devices.alice: devices.alice(lambda: x[:, :15])(),
            devices.bob: devices.bob(lambda: x[:, 15:])(),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label_data = FedNdarray(
        partitions={devices.alice: devices.alice(lambda: y)()},
        partition_way=PartitionWay.VERTICAL,
    )

    _wait_io([v_data, label_data])
    logging.info(f"IO times: {time.time() - start}s")

    _run_test(
        devices,
        "breast_cancer",
        v_data,
        label_data,
        y,
        128,
        'Logit',
        'Bernoulli',
        l2_lambda=1.0,
    )


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_gamma_data(sf_production_setup_devices):
    devices = sf_production_setup_devices

    start = time.time()
    import statsmodels.api as sm

    data = sm.datasets.scotland.load()

    x = _transform(np.array(data.exog))
    y = np.array(data.endog)

    v_data = FedNdarray(
        partitions={
            devices.alice: devices.alice(lambda: x[:, :3])(),
            devices.bob: devices.bob(lambda: x[:, 3:])(),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label_data = FedNdarray(
        partitions={devices.alice: devices.alice(lambda: y)()},
        partition_way=PartitionWay.VERTICAL,
    )

    _wait_io([v_data, label_data])
    logging.info(f"IO times: {time.time() - start}s")

    _run_test(
        devices,
        "gamma",
        v_data,
        label_data,
        y,
        32,
        'Log',
        'Gamma',
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
#     #    python tests/ml/linear/test_ss_glm.py

#     # !!!!!!
#     # This example contains two test: irls mode glm and sgd mode glm
#     # need to be timed separately in benchmark testing.
#     # !!!!!!

#     # <<< !!! >>> uncomment next line if you need run this demo under MPU.
#     # jax.config.update("jax_enable_x64", True)

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
#         # split 1-11 columns to alice and 11-21 columns to bob which include y col.
#         # <<< !!! >>> replace parts range to your own dataset's columns count.
#         parts={cluster.alice: (1, 11), cluster.bob: (11, 22)},
#         # split by vertical. DON'T change this.
#         axis=1,
#     )
#     # split y out of dataset,
#     # <<< !!! >>> change 'y' if label column name is not y in dataset.
#     label_data = vdf["y"]
#     # v_data remains all features.
#     v_data = vdf.drop(columns="y")
#     # <<< !!! >>> change cluster.bob if y not belong to bob.
#     y = reveal(label_data.partitions[cluster.bob].data)
#     wait([p.data for p in v_data.partitions.values()])
#     logging.info(f"IO times: {time.time() - start}s")

#     # <<< !!! >>> run irls mode glm
#     model = SSGLM(cluster.spu)
#     start = time.time()
#     model.fit_irls(
#         v_data,
#         label_data,
#         None,
#         None,
#         3,
#         'Logit',
#         'Bernoulli',
#         1,
#         1,
#     )
#     logging.info(f"main irls mode train time: {time.time() - start}")
#     start = time.time()
#     spu_yhat = model.predict(v_data)
#     yhat = reveal(spu_yhat)
#     logging.info(f"main predict time: {time.time() - start}")
#     logging.info(f"main auc: {roc_auc_score(y, yhat)}")

#     # <<< !!! >>> run sgd mode glm
#     start = time.time()
#     model.fit_sgd(
#         v_data,
#         label_data,
#         None,
#         None,
#         3,
#         'Logit',
#         'Bernoulli',
#         1,
#         1,
#         0.3,
#         iter_start_irls=0,
#         batch_size=128,
#     )
#     logging.info(f"main sgd mode train time: {time.time() - start}")
#     start = time.time()
#     spu_yhat = model.predict(v_data)
#     yhat = reveal(spu_yhat)
#     logging.info(f"main predict time: {time.time() - start}")
#     logging.info(f"main auc: {roc_auc_score(y, yhat)}")
