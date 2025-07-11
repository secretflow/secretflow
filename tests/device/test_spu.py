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

import tempfile

import jax
import numpy as np
import pytest
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, Relu

import secretflow as sf
from secretflow.device.device.spu import SPUObject


def MLP():
    nn_init, nn_apply = stax.serial(
        Dense(30),
        Relu,
        Dense(15),
        Relu,
        Dense(8),
        Relu,
        Dense(1),
    )

    return nn_init, nn_apply


def init_state(learning_rate):
    KEY = jax.random.PRNGKey(42)
    INPUT_SHAPE = (-1, 30)

    init_fun, predict_fun = MLP()
    _, params_init = init_fun(KEY, INPUT_SHAPE)
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_state = opt_init(params_init)
    return opt_state


def _test_scalar(devices):
    x = devices.alice(lambda: 1)()
    x_ = x.to(devices.spu)
    assert x_.device == devices.spu
    y = x_.to(devices.bob)
    np.testing.assert_almost_equal(sf.reveal(x), sf.reveal(y), decimal=5)


@pytest.mark.mpc
def test_scalar_prod(sf_production_setup_devices):
    _test_scalar(sf_production_setup_devices)


def test_scalar_sim(sf_simulation_setup_devices):
    _test_scalar(sf_simulation_setup_devices)


def _test_ndarray(devices):
    x = devices.alice(np.random.uniform)(-10, 10, (3, 4))
    x_ = x.to(devices.spu)
    assert x_.device == devices.spu
    y = x_.to(devices.bob)
    np.testing.assert_almost_equal(sf.reveal(x), sf.reveal(y), decimal=5)


@pytest.mark.mpc
def test_ndarray_prod(sf_production_setup_devices):
    _test_ndarray(sf_production_setup_devices)


def test_ndarray_sim(sf_simulation_setup_devices):
    _test_ndarray(sf_simulation_setup_devices)


def _test_pytree(devices):
    x = devices.alice(
        lambda: [
            [np.random.rand(3, 4), np.random.rand(4, 5)],
            {'weights': [1.0, 2.0]},
        ]
    )()
    x_ = x.to(devices.spu)
    assert x_.device == devices.spu
    y = x_.to(devices.bob)

    expected, actual = sf.reveal(x), sf.reveal(y)
    expected_flat, expected_tree = jax.tree_util.tree_flatten(expected)
    actual_flat, actual_tree = jax.tree_util.tree_flatten(actual)

    assert expected_tree == actual_tree
    assert len(expected_flat) == len(actual_flat)
    for expected, actual in zip(expected_flat, actual_flat):
        np.testing.assert_almost_equal(expected, actual, decimal=5)


@pytest.mark.mpc
def test_pytree_prod(sf_production_setup_devices):
    _test_pytree(sf_production_setup_devices)


def test_pytree_sim(sf_simulation_setup_devices):
    _test_pytree(sf_simulation_setup_devices)


def _test_to_heu(devices):
    x = devices.alice(np.random.uniform)(-10, 10, (30, 40))
    x_spu = x.to(devices.spu)

    # spu -> heu
    x_heu = x_spu.to(devices.heu)
    y = x_heu.to(devices.alice)
    np.testing.assert_almost_equal(sf.reveal(x), sf.reveal(y), decimal=5)

    # heu -> spu
    x_spu = x_heu.to(devices.spu)
    y = x_spu.to(devices.alice)
    np.testing.assert_almost_equal(sf.reveal(x), sf.reveal(y), decimal=5)


@pytest.mark.mpc
def test_to_heu_prod(sf_production_setup_devices):
    _test_to_heu(sf_production_setup_devices)


def test_to_heu_sim(sf_simulation_setup_devices):
    _test_to_heu(sf_simulation_setup_devices)


def _test_dump_load(devices):
    world_size = devices.spu.world_size

    if world_size == 2:
        _, alice_path = tempfile.mkstemp()
        _, bob_path = tempfile.mkstemp()
        paths = [alice_path, bob_path]
    elif world_size == 3:
        _, alice_path = tempfile.mkstemp()
        _, bob_path = tempfile.mkstemp()
        _, carol_path = tempfile.mkstemp()
        paths = [alice_path, bob_path, carol_path]

    x = devices.alice(np.random.uniform)(-10, 10, (3, 4))
    x_spu = x.to(devices.spu)

    devices.spu.dump(x_spu, paths)

    x_spu_ = devices.spu.load(paths)
    assert isinstance(x_spu_, SPUObject)
    np.testing.assert_almost_equal(sf.reveal(x_spu), sf.reveal(x_spu_), decimal=5)


@pytest.mark.mpc
def test_dump_load_prod(sf_production_setup_devices):
    _test_dump_load(sf_production_setup_devices)


def test_dump_load_sim(sf_simulation_setup_devices):
    _test_dump_load(sf_simulation_setup_devices)
