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
from sklearn.datasets import make_blobs

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ml.cluster.kmeans import KMeans
from tests.sf_fixtures import SFProdParams


def _load_data():
    n_samples = 1000
    n_features = 100
    np.random.seed(0)
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=2)
    split_index = n_features // 2
    return X[:, :split_index], X[:, split_index:]


def _run_test(devices, test_name, v_data, n_clusters, init, max_iter, concate_data):
    model = KMeans(devices.spu)

    start = time.time()
    model.fit(v_data, n_clusters=n_clusters, init=init, max_iter=max_iter)
    logging.info(f"{test_name} kmeans fit time: {time.time() - start}")

    start = time.time()
    spu_result = model.predict(v_data)
    result = reveal(spu_result)
    logging.info(f"{test_name} predict time: {time.time() - start}")

    # Compare with sklearn
    from sklearn.cluster import KMeans as SL_KMeans

    expect_model = SL_KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter)
    except_res = expect_model.fit(concate_data).predict(concate_data)

    logging.info(
        f"{test_name} sml centers: {reveal(devices.spu(lambda model: model._centers)(model.model))}"
    )
    logging.info(f"{test_name} sklearn centers: {expect_model.cluster_centers_}")
    logging.info(f"{test_name} predict result: {result}")
    logging.info(f"{test_name} sklearn predict result: {except_res}")

    assert (
        result.shape[0] == except_res.shape[0]
    ), f"{result.shape} == {except_res.shape}"
    if result[0] == except_res[0]:
        assert jnp.array_equal(result, except_res)
    else:
        assert jnp.array_equal(result, except_res ^ 1)


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
    x1, x2 = _load_data()
    x1 = jnp.array([[100], [100], [-100], [-100]])
    x2 = jnp.array([[-100], [-100], [100], [100]])
    v_data, concate_data = get_vdata_and_concate_data(
        sf_production_setup_devices, x1, x2
    )

    _run_test(
        sf_production_setup_devices,
        test_name="work",
        v_data=v_data,
        n_clusters=2,
        init="random",
        max_iter=10,
        concate_data=concate_data,
    )
