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
from heu import phe

import secretflow.device as ft
from secretflow import reveal
from secretflow.device.device.heu import HEUMoveConfig


def _test_device(devices):
    x = ft.with_device(devices.alice)(np.random.rand)(3, 4)
    x_ = x.to(devices.heu)
    assert x_.device == devices.heu

    # Can't place a user-defined function to HEU Device
    with pytest.raises(NotImplementedError):

        @ft.with_device(devices.heu)
        def add(a, b):
            return a + b

        y = add(x_, x_)

    # Can't convert an HEUTensor to CPUTensor without secret key
    with pytest.raises(AssertionError):
        y = x_.to(devices.bob)

    y = x_.to(devices.alice)
    assert y.device == devices.alice
    np.testing.assert_almost_equal(reveal(x), reveal(y), decimal=4)


@pytest.mark.mpc
def test_device_prod(sf_production_setup_devices):
    _test_device(sf_production_setup_devices)


def test_device_sim(sf_simulation_setup_devices):
    _test_device(sf_simulation_setup_devices)


def _test_math_ops(devices):
    schema = phe.SchemaType.ZPaillier
    x = ft.with_device(devices.alice)(np.random.rand)(3, 4)
    y = ft.with_device(devices.bob)(np.random.rand)(3, 4)
    y_int = ft.with_device(devices.bob)(np.random.randint)(10, size=(3, 4))
    z_int = ft.with_device(devices.bob)(np.random.randint)(10, size=(4, 5))

    x_, y_, y_int_, z_int_ = (
        x.to(devices.heu),  # x_ is ciphertext
        y.to(devices.heu),
        y_int.to(
            devices.heu, config=HEUMoveConfig(heu_encoder=phe.BigintEncoder(schema))
        ),
        z_int.to(
            devices.heu, config=HEUMoveConfig(heu_encoder=phe.BigintEncoder(schema))
        ),
    )  # plaintext

    add_ = x_ + y_  # shape: 3x4
    sub_ = x_ - y_  # shape: 3x4
    # x_ is 1x scaled, y_int_ is not scaled, so result is 1x scaled
    mul_ = x_ * y_int_  # shape: 3x4
    matmul_ = x_ @ z_int_  # shape: 3x5

    x, y, y_int, z_int = reveal(x), reveal(y), reveal(y_int), reveal(z_int)
    add, sub = add_.to(devices.alice), sub_.to(devices.alice)
    mul = mul_.to(devices.alice)
    matmul = matmul_.to(devices.alice)

    np.testing.assert_almost_equal(x + y, reveal(add), decimal=4)
    np.testing.assert_almost_equal(x - y, reveal(sub), decimal=4)
    np.testing.assert_almost_equal(x * y_int, reveal(mul), decimal=4)
    np.testing.assert_almost_equal(x @ z_int, reveal(matmul), decimal=4)

    # test slice
    add = reveal(add_[2, 3].to(devices.alice))
    sub = reveal(sub_[1:3, :].to(devices.alice))
    mul = reveal(mul_[:3:2, ::-1].to(devices.alice))
    matmul = reveal(matmul_[[0, 1, 2], 1::2].to(devices.alice))

    assert isinstance(add, float)  # add is scalar
    assert sub.shape == (2, 4)
    assert mul.shape == (2, 4)
    assert matmul.shape == (3, 2)
    np.testing.assert_almost_equal((x + y)[2, 3], add, decimal=4)
    np.testing.assert_almost_equal((x - y)[1:3, :], sub, decimal=4)
    np.testing.assert_almost_equal((x * y_int)[:3:2, ::-1], mul, decimal=4)
    np.testing.assert_almost_equal((x @ z_int)[[0, 1, 2], 1::2], matmul, decimal=4)


@pytest.mark.mpc
def test_math_ops_prod(sf_production_setup_devices):
    _test_math_ops(sf_production_setup_devices)


def test_math_ops_sim(sf_simulation_setup_devices):
    _test_math_ops(sf_simulation_setup_devices)


def _test_sum(devices):
    # test vector, ciphertext
    m = ft.with_device(devices.alice)(np.random.rand)(20)
    m_heu = m.to(devices.heu)  # ciphertext
    np.testing.assert_almost_equal(reveal(m).sum(), reveal(m_heu.sum()), decimal=4)
    np.testing.assert_almost_equal(
        reveal(m)[[1, 2, 3]].sum(), reveal(m_heu[[1, 2, 3]].sum()), decimal=4
    )
    np.testing.assert_almost_equal(
        reveal(m)[3:10].sum(), reveal(m_heu[3:10].sum()), decimal=4
    )

    # test matrix
    m = ft.with_device(devices.bob)(np.random.rand)(20, 20)
    m_heu = m.to(
        devices.heu, HEUMoveConfig(heu_dest_party=devices.bob.party)
    )  # plaintext
    assert m_heu.is_plain
    np.testing.assert_almost_equal(reveal(m).sum(), reveal(m_heu.sum()), decimal=4)
    np.testing.assert_almost_equal(
        reveal(m).sum(), reveal(m_heu.encrypt().sum()), decimal=4
    )


@pytest.mark.mpc
def test_sum_prod(sf_production_setup_devices):
    _test_sum(sf_production_setup_devices)


def test_sum_sim(sf_simulation_setup_devices):
    _test_sum(sf_simulation_setup_devices)
