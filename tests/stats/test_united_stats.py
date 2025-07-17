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

import numpy as np
import pytest

from secretflow import reveal
from secretflow.stats.united_stats import (
    united_mean,
    united_mean_and_var,
    united_median,
    united_var,
)
from tests.sf_fixtures import mpc_fixture

TEST_SIZE = 2000


@mpc_fixture
def prod_env_and_data(sf_production_setup_devices):
    np.random.seed(22)
    y_true = np.round(np.random.random((TEST_SIZE,)).reshape(-1))
    y_pred = np.random.random((TEST_SIZE,)).reshape(-1)
    yield sf_production_setup_devices, [
        sf_production_setup_devices.alice(lambda x: x)(y_true),
        sf_production_setup_devices.bob(lambda x: x)(y_pred),
        np.concatenate([y_true, y_pred]),
    ]


@mpc_fixture
def prod_env_and_data_3_party(sf_production_setup_devices):
    np.random.seed(22)
    y_true = np.round(np.random.random((TEST_SIZE,)).reshape(-1))
    y_pred = np.random.random((TEST_SIZE,)).reshape(-1)
    y_additional = np.random.random((TEST_SIZE,)).reshape(-1)
    yield sf_production_setup_devices, [
        sf_production_setup_devices.alice(lambda x: x)(y_true),
        sf_production_setup_devices.bob(lambda x: x)(y_pred),
        sf_production_setup_devices.carol(lambda x: x)(y_additional),
        np.concatenate([y_true, y_pred, y_additional]),
    ]


@pytest.mark.mpc
def test_united_mean(prod_env_and_data):
    env, data = prod_env_and_data
    y_true, y_pred, y_total = data[0], data[1], data[2]
    true_mean = y_total.mean()
    united_mean_result = reveal(united_mean([y_true, y_pred], env.spu))
    np.testing.assert_almost_equal(true_mean, united_mean_result, decimal=5)


@pytest.mark.mpc
def test_united_mean_and_var(prod_env_and_data):
    env, data = prod_env_and_data
    y_true, y_pred, y_total = data[0], data[1], data[2]
    true_mean = y_total.mean()
    true_var = y_total.var()
    mean_val, var_val = reveal(united_mean_and_var([y_true, y_pred], env.spu))
    np.testing.assert_almost_equal(true_mean, mean_val, decimal=5)
    np.testing.assert_almost_equal(true_var, var_val, decimal=3)


@pytest.mark.mpc
def test_united_var(prod_env_and_data):
    env, data = prod_env_and_data
    y_true, y_pred, y_total = data[0], data[1], data[2]
    true_var = y_total.var()
    var_val = reveal(united_var([y_true, y_pred], env.spu, block_size=1000))
    np.testing.assert_almost_equal(true_var, var_val, decimal=3)


@pytest.mark.mpc
def test_united_median(prod_env_and_data):
    env, data = prod_env_and_data
    y_true, y_pred, y_total = data[0], data[1], data[2]
    true_med = np.median(y_total)
    med_val = reveal(united_median([y_true, y_pred], env.spu))
    np.testing.assert_almost_equal(true_med, med_val, decimal=6)


@pytest.mark.mpc(parties=3)
def test_united_median_3_party(prod_env_and_data_3_party):
    env, data = prod_env_and_data_3_party
    y_true, y_pred, y_additional, y_total = data[0], data[1], data[2], data[3]
    true_med = np.median(y_total)
    med_val = reveal(united_median([y_true, y_pred, y_additional], env.spu))
    np.testing.assert_almost_equal(true_med, med_val, decimal=6)
