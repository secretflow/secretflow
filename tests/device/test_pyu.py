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

import secretflow.device as ft
from secretflow import reveal
from secretflow.device.device.pyu import PYUObject
from secretflow.device.device.spu import SPUObject


def _test_device(devices):
    @ft.with_device(devices.alice)
    def load(*shape):
        return np.random.rand(*shape)

    x = load(3, 4)
    y = x.to(devices.bob)

    assert y.device == devices.bob
    np.testing.assert_equal(reveal(x), reveal(y))


@pytest.mark.mpc
def test_device_prod(sf_production_setup_devices):
    _test_device(sf_production_setup_devices)


def test_device_sim(sf_simulation_setup_devices):
    _test_device(sf_simulation_setup_devices)


def _test_average(devices):
    def average(*a, axis=None, weights=None):
        return np.average(a, axis=axis, weights=weights)

    x = ft.with_device(devices.alice)(np.random.rand)(3, 4)
    y = ft.with_device(devices.bob)(np.random.rand)(3, 4)

    with pytest.raises(AssertionError):
        devices.alice(average)(x, y, axis=0)

    y = y.to(devices.alice)
    actual = devices.alice(average)(x, y, axis=0, weights=(1, 2))
    expected = np.average([reveal(x), reveal(y)], axis=0, weights=(1, 2))
    np.testing.assert_equal(reveal(actual), expected)


@pytest.mark.mpc
def test_average_prod(sf_production_setup_devices):
    _test_average(sf_production_setup_devices)


def test_average_sim(sf_simulation_setup_devices):
    _test_average(sf_simulation_setup_devices)


def _test_multiple_return(devices):
    def load():
        return 1, 2, 3

    x, y, z = devices.alice(load, num_returns=3)()
    assert isinstance(x, ft.PYUObject)

    x, y, z = ft.reveal([x, y, z])
    assert x == 1


@pytest.mark.mpc
def test_multiple_return_prod(sf_production_setup_devices):
    _test_multiple_return(sf_production_setup_devices)


def test_multiple_return_sim(sf_simulation_setup_devices):
    _test_multiple_return(sf_simulation_setup_devices)


def _test_dictionary_return(devices):
    def load():
        return {'a': 1, 'b': 23}

    x = devices.alice(load)()
    assert isinstance(x, ft.PYUObject)
    assert ft.reveal(x) == {'a': 1, 'b': 23}

    x_ = x.to(devices.spu)
    assert ft.reveal(x_) == {'a': 1, 'b': 23}


@pytest.mark.mpc
def test_dictionary_return_prod(sf_production_setup_devices):
    _test_dictionary_return(sf_production_setup_devices)


def test_dictionary_return_sim(sf_simulation_setup_devices):
    _test_dictionary_return(sf_simulation_setup_devices)


def _test_to(devices):
    @ft.with_device(devices.alice)
    def load(*shape):
        return np.random.rand(*shape)

    x = load(3, 4)
    assert isinstance(x, PYUObject)

    x_1 = x.to(devices.spu)
    assert isinstance(x_1, SPUObject)
    assert np.allclose(ft.reveal(x), ft.reveal(x_1))


@pytest.mark.mpc
def test_to_prod(sf_production_setup_devices):
    _test_to(sf_production_setup_devices)


def test_to_sim(sf_simulation_setup_devices):
    _test_to(sf_simulation_setup_devices)


def _test_io(devices):
    def load():
        return {'a': 1, 'b': 23}

    x = devices.alice(load)()

    import tempfile

    _, path = tempfile.mkstemp()
    ft.wait(devices.alice.dump(x, path))
    x_ = devices.alice.load(path)
    # self.assertTrue(isinstance(x_, PYUObject))
    assert isinstance(x_, PYUObject)
    # self.assertEqual(ft.reveal(x_), {'a': 1, 'b': 23})
    assert ft.reveal(x_) == {'a': 1, 'b': 23}


@pytest.mark.mpc
def test_io_prod(sf_production_setup_devices):
    _test_io(sf_production_setup_devices)


def test_io_sim(sf_simulation_setup_devices):
    _test_io(sf_simulation_setup_devices)
