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

from typing import Tuple

import numpy as np
import pytest

from secretflow import reveal
from secretflow.device import PYUObject, proxy
from tests.sf_fixtures import mpc_fixture


@proxy(PYUObject)
class Model:
    def __init__(self, builder):
        self.weights = builder()
        self.dataset_x = None
        self.dataset_y = None

    def build_dataset(self, x, y):
        self.dataset_x = x
        self.dataset_y = y

    def retrieve_dataset(self):
        return self.dataset_x, self.dataset_y

    def get_weights(self):
        return self.weights

    def compute_weights(self) -> Tuple[np.ndarray, int]:
        return self.weights, 100


@mpc_fixture
def prod_env_and_model(sf_production_setup_devices):
    model = Model(lambda: np.ones((3, 4)), device=sf_production_setup_devices.alice)
    return sf_production_setup_devices, model


@pytest.fixture
def sim_env_and_model(sf_simulation_setup_devices):
    model = Model(lambda: np.ones((3, 4)), device=sf_simulation_setup_devices.alice)
    yield sf_simulation_setup_devices, model


def _test_init_without_device(devices):
    with pytest.raises(AssertionError, match='missing device argument'):
        Model(lambda: np.ones((3, 4)))


@pytest.mark.mpc
def test_init_without_device_prod(sf_production_setup_devices):
    _test_init_without_device(sf_production_setup_devices)


def test_init_without_device_sim(sf_simulation_setup_devices):
    _test_init_without_device(sf_simulation_setup_devices)


def _test_init_with_mismatch_device(devices):
    with pytest.raises(AssertionError, match='unexpected device type'):
        Model(lambda: np.ones((3, 4)), device=devices.spu)


@pytest.mark.mpc
def test_init_with_mismatch_device_prod(sf_production_setup_devices):
    _test_init_with_mismatch_device(sf_production_setup_devices)


def test_init_with_mismatch_device_sim(sf_simulation_setup_devices):
    _test_init_with_mismatch_device(sf_simulation_setup_devices)


def _test_call_with_mismatch_device(devices, model):
    x, y = devices.alice(np.random.rand)(3, 4), devices.bob(np.random.rand)(3)
    with pytest.raises(AssertionError, match='unexpected device object'):
        model.build_dataset(x, y)


@pytest.mark.mpc
def test_call_with_mismatch_device_prod(prod_env_and_model):
    env, model = prod_env_and_model
    _test_call_with_mismatch_device(env, model)


def test_call_with_mismatch_device_sim(sim_env_and_model):
    env, model = sim_env_and_model
    _test_call_with_mismatch_device(env, model)


def _test_single_return(devices, model):
    weights = model.get_weights()
    assert weights.device == devices.alice
    weights = reveal(weights)
    np.testing.assert_equal(weights, np.ones((3, 4)))


@pytest.mark.mpc
def test_single_return_prod(prod_env_and_model):
    env, model = prod_env_and_model
    _test_single_return(env, model)


def test_single_return_sim(sim_env_and_model):
    env, model = sim_env_and_model
    _test_single_return(env, model)


def _test_multiple_return(devices, model):
    weights, n = model.compute_weights()
    assert weights.device == devices.alice
    assert n.device == devices.alice

    weights, n = reveal(weights), reveal(n)
    np.testing.assert_equal(weights, np.ones((3, 4)))
    assert n == 100


@pytest.mark.mpc
def test_multiple_return_prod(prod_env_and_model):
    env, model = prod_env_and_model
    _test_multiple_return(env, model)


def test_multiple_return_sim(sim_env_and_model):
    env, model = sim_env_and_model
    _test_multiple_return(env, model)


def _test_multiple_return_without_annotation(devices, model):
    x, y = devices.alice(np.random.rand)(3, 4), devices.alice(np.random.rand)(3)
    model.build_dataset(x, y)

    res = model.retrieve_dataset()
    assert res.device == devices.alice

    x, y = reveal([x, y])
    x_, y_ = reveal(res)
    np.testing.assert_equal(x_, x)
    np.testing.assert_equal(y_, y)


@pytest.mark.mpc
def test_multiple_return_without_annotation_prod(prod_env_and_model):
    env, model = prod_env_and_model
    _test_multiple_return_without_annotation(env, model)


def test_multiple_return_without_annotation_sim(sim_env_and_model):
    env, model = sim_env_and_model
    _test_multiple_return_without_annotation(env, model)
