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
import numpy as np
import pytest
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.neighbors import KNeighborsClassifier

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.naive_bayes import GNB
from tests.sf_fixtures import SFProdParams


def _load_data():
    # 假设有一组测试样本
    # Create a simple dataset
    partial = 0.5
    n_samples = 1000
    n_features = 100
    centers = 3
    np.random.seed(0)
    X, y = datasets.make_blobs(
        n_samples=n_samples, n_features=n_features, centers=centers
    )
    classes = jnp.unique(y)
    assert len(classes) == centers, f'Retry or increase partial.'
    split_idx = int(partial * len(y))
    feature_split = n_features // 2

    train = (
        X[:split_idx, :feature_split],
        X[:split_idx, feature_split:],
        y[:split_idx],
    )
    test = (
        X[split_idx:, :feature_split],
        X[split_idx:, feature_split:],
        y[split_idx:],
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
    train_vdata, X = get_vdata_and_concate_data(sf_production_setup_devices, x1, x2)
    train_label = FedNdarray(
        partitions={
            sf_production_setup_devices.alice: sf_production_setup_devices.alice(
                lambda: y
            )(),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    test_vdata, tx = get_vdata_and_concate_data(sf_production_setup_devices, x1t, x2t)

    model = GNB(sf_production_setup_devices.spu)

    start = time.time()
    model.fit(train_vdata, train_label, n_classes=3)
    theta, var = reveal(
        sf_production_setup_devices.spu(lambda m: (m.theta_, m.var_))(model.model)
    )
    logging.info(f" GNB fit time: {time.time() - start}")

    start = time.time()
    spu_result = model.predict(test_vdata)
    result = reveal(spu_result)
    logging.info(f"predict time: {time.time() - start}")
    logging.info(f"predict result: {result.shape}")
    logging.info(f"sml predict result: {(result == yt).sum() / len(yt)}")

    skl_gnb = SklearnGaussianNB()
    skl_gnb.fit(X, y)
    skl_theta = skl_gnb.theta_
    skl_var = skl_gnb.var_
    sklearn_predictions = skl_gnb.predict(tx)
    logging.info(f"skl predict result: {(sklearn_predictions == yt).sum() / len(yt)}")

    assert jnp.array_equal(
        result, sklearn_predictions
    ), f"sml: {result}, skl: {sklearn_predictions}"
    assert np.allclose(theta, skl_theta, rtol=1.0e-5, atol=1)
    assert np.allclose(var, skl_var, rtol=1.0e-5, atol=1)
