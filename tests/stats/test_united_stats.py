import numpy as np
import pytest

from secretflow import reveal
from secretflow.stats.united_stats import (
    united_mean,
    united_mean_and_var,
    united_median,
    united_var,
)

TEST_SIZE = 2000


@pytest.fixture(scope='module')
def prod_env_and_data(sf_production_setup_devices):
    np.random.seed(22)
    y_true = np.round(np.random.random((TEST_SIZE,)).reshape(-1))
    y_pred = np.random.random((TEST_SIZE,)).reshape(-1)
    yield sf_production_setup_devices, [
        sf_production_setup_devices.alice(lambda x: x)(y_true),
        sf_production_setup_devices.bob(lambda x: x)(y_pred),
        np.concatenate([y_true, y_pred]),
    ]


def test_united_mean(prod_env_and_data):
    env, data = prod_env_and_data
    y_true, y_pred, y_total = data[0], data[1], data[2]
    true_mean = y_total.mean()
    united_mean_result = reveal(united_mean([y_true, y_pred], env.spu))
    np.testing.assert_almost_equal(true_mean, united_mean_result, decimal=5)


def test_united_mean_and_var(prod_env_and_data):
    env, data = prod_env_and_data
    y_true, y_pred, y_total = data[0], data[1], data[2]
    true_mean = y_total.mean()
    true_var = y_total.var()
    mean_val, var_val = reveal(united_mean_and_var([y_true, y_pred], env.spu))
    np.testing.assert_almost_equal(true_mean, mean_val, decimal=5)
    np.testing.assert_almost_equal(true_var, var_val, decimal=3)


def test_united_var(prod_env_and_data):
    env, data = prod_env_and_data
    y_true, y_pred, y_total = data[0], data[1], data[2]
    true_var = y_total.var()
    var_val = reveal(united_var([y_true, y_pred], env.spu, block_size=1000))
    np.testing.assert_almost_equal(true_var, var_val, decimal=3)


def test_united_median(prod_env_and_data):
    env, data = prod_env_and_data
    y_true, y_pred, y_total = data[0], data[1], data[2]
    true_med = np.median(y_total)
    med_val = reveal(united_median([y_true, y_pred], env.spu))
    np.testing.assert_almost_equal(true_med, med_val, decimal=6)
