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
import time

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ml.linear import SSRegression
from secretflow.utils.simulation.datasets import load_linear


def _transform(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def _wait_io(inputs):
    wait_objs = list()
    for input in inputs:
        wait_objs.extend([input.partitions[d] for d in input.partitions])
    wait(wait_objs)


def _run_test(devices, test_name, v_data, label_data, y, batch_size):
    reg = SSRegression(devices.spu)
    start = time.time()
    reg.fit(
        v_data,
        label_data,
        3,
        0.3,
        batch_size,
        "t1",
        "logistic",
        "l2",
        0.5,
    )
    logging.info(f"{test_name} train time: {time.time() - start}")
    start = time.time()
    spu_yhat = reg.predict(v_data, batch_size)
    yhat = reveal(spu_yhat)
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    logging.info(f"{test_name} predict time: {time.time() - start}")
    logging.info(f"{test_name} auc: {roc_auc_score(y, yhat)}")

    fed_yhat = reg.predict(v_data, batch_size, devices.alice)
    assert len(fed_yhat.partitions) == 1 and devices.alice in fed_yhat.partitions
    yhat = reveal(fed_yhat.partitions[devices.alice])
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    logging.info(f"{test_name} auc: {roc_auc_score(y, yhat)}")

    start = time.time()
    reg.fit(
        v_data,
        label_data,
        3,
        0.1,
        batch_size,
        "t1",
        "logistic",
        "l2",
        0.05,
        decay_epoch=2,
        decay_rate=0.5,
        strategy="policy_sgd",
    )
    logging.info(f"{test_name} policy-sgd train time: {time.time() - start}")
    start = time.time()

    spu_yhat = reg.predict(v_data, batch_size)
    yhat = reveal(spu_yhat)
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    logging.info(f"{test_name} predict time: {time.time() - start}")
    logging.info(f"{test_name} auc: {roc_auc_score(y, yhat)}")

    fed_yhat = reg.predict(v_data, batch_size, devices.alice)
    assert len(fed_yhat.partitions) == 1 and devices.alice in fed_yhat.partitions
    yhat = reveal(fed_yhat.partitions[devices.alice])
    assert yhat.shape[0] == y.shape[0], f"{yhat.shape} == {y.shape}"
    logging.info(f"{test_name} auc: {roc_auc_score(y, yhat)}")


def test_breast_cancer(sf_production_setup_devices_aby3):
    from sklearn.datasets import load_breast_cancer

    start = time.time()
    ds = load_breast_cancer()
    x, y = _transform(ds["data"]), ds["target"]

    v_data = FedNdarray(
        partitions={
            sf_production_setup_devices_aby3.alice: sf_production_setup_devices_aby3.alice(
                lambda: x[:, :15]
            )(),
            sf_production_setup_devices_aby3.bob: sf_production_setup_devices_aby3.bob(
                lambda: x[:, 15:]
            )(),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label_data = FedNdarray(
        partitions={
            sf_production_setup_devices_aby3.alice: sf_production_setup_devices_aby3.alice(
                lambda: y
            )()
        },
        partition_way=PartitionWay.VERTICAL,
    )

    _wait_io([v_data, label_data])
    logging.info(f"IO times: {time.time() - start}s")

    _run_test(
        sf_production_setup_devices_aby3, "breast_cancer", v_data, label_data, y, 128
    )


def test_linear(sf_production_setup_devices_aby3):
    start = time.time()
    vdf = load_linear(
        parts={
            sf_production_setup_devices_aby3.alice: (1, 11),
            sf_production_setup_devices_aby3.bob: (11, 22),
        }
    )
    label_data = vdf["y"]
    v_data = vdf.drop(columns="y")
    y = reveal(label_data.partitions[sf_production_setup_devices_aby3.bob].data)
    _wait_io([v_data.values, label_data.values])
    logging.info(f"IO times: {time.time() - start}s")

    _run_test(sf_production_setup_devices_aby3, "linear", v_data, label_data, y, 1024)


# TODO(fengjun.feng): move the following to a seperate integration test.

# if __name__ == '__main__':
#     # HOW TO RUN:
#     # 0. change args following <<< !!! >>> flag.
#     #    you need change input data path & train settings before run.
#     # 1. install requirements following INSTALLATION.md
#     # 2. set env
#     #    export PYTHONPATH=$PYTHONPATH:bazel-bin
#     # 3. run
#     #    python tests/ml/linear/test_ss_lr.py

#     # use aby3 in this example.
#     cluster = ABY3MultiDriverDeviceTestCase()
#     cluster.setUpClass()
#     # init log
#     logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#     # prepare data
#     start = time.time()
#     # read dataset.
#     vdf = create_df(
#         # load file 'dataset('linear')' as train dataset.
#         # <<< !!! >>> replace dataset path to your own local file.
#         dataset('linear'),
#         # split 1-10 columns to alice and 11-21 columns to bob which include y col.
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

#     # run ss lr
#     reg = SSRegression(cluster.spu)
#     start = time.time()
#     reg.fit(
#         v_data,
#         label_data,
#         # <<< !!! >>> change args to your test settings.
#         3,  # epochs
#         0.3,
#         1024,  # batch_size
#         't1',  # sig_type
#         'logistic',  # reg_type
#         'l2',  # penalty
#         0.5,
#     )
#     logging.info(f"main train time: {time.time() - start}")
#     start = time.time()
#     spu_yhat = reg.predict(v_data, 1024)
#     yhat = reveal(spu_yhat)
#     logging.info(f"main predict time: {time.time() - start}")
#     logging.info(f"main auc: {roc_auc_score(y, yhat)}")
