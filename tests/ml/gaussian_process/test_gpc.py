# Copyright 2025 Ant Group Co., Ltd.
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

import jax.numpy as jnp
import pytest
from sklearn.datasets import load_iris

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.gaussian_process import GPC
from tests.sf_fixtures import SFProdParams


def _load_data():
    X, y = load_iris(return_X_y=True)

    train_ids = list(range(45, 55)) + list(range(100, 105))
    test_ids = list(range(0, 5)) + list(range(55, 60)) + list(range(110, 115))

    feature_split = int(0.5 * X.shape[1])
    train = (
        X[train_ids, :feature_split],
        X[train_ids, feature_split:],
        y[train_ids],
    )
    test = (
        X[test_ids, :feature_split],
        X[test_ids, feature_split:],
        y[test_ids],
    )

    return train, test


def get_vdata_and_concate_data(devices, x1, x2):
    vdata = FedNdarray(
        partitions={
            devices.alice: devices.alice(lambda: x1)(),
            devices.bob: devices.bob(lambda: x2)(),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    concate_data = jnp.concatenate((x1, x2), axis=1)

    return vdata, concate_data


@pytest.mark.mpc(parties=3, params=SFProdParams.ABY3)
def test_work(sf_production_setup_devices):
    train, test = _load_data()
    x1, x2, y = train
    x1t, x2t, yt = test
    train_vdata, _ = get_vdata_and_concate_data(sf_production_setup_devices, x1, x2)
    train_label = FedNdarray(
        partitions={
            sf_production_setup_devices.alice: sf_production_setup_devices.alice(
                lambda: y
            )(),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    test_vdata, _ = get_vdata_and_concate_data(sf_production_setup_devices, x1t, x2t)

    model = GPC(sf_production_setup_devices.spu)

    start = time.time()
    model.fit(train_vdata, train_label, max_iter_predict=20, n_classes=3)
    logging.info(f" GPC fit time: {time.time() - start}")

    start = time.time()
    spu_result = model.predict(test_vdata)
    result = reveal(spu_result)
    logging.info(f"predict time: {time.time() - start}")

    assert jnp.array_equal(result, yt)
