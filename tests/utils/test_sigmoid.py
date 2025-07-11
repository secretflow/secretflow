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

from secretflow.device.driver import reveal
from secretflow.utils import sigmoid


def get_spu_x(env, x):
    pyu = env.alice(lambda: x)()
    return pyu.to(env.spu)


def do_test(env, fn):
    x_ = env.alice(lambda: np.random.normal(0, 5, size=(5, 5)))()
    x = reveal(x_)
    spu = reveal(env.spu(fn)(x_.to(env.spu)))
    jnp = fn(x)
    np.testing.assert_almost_equal(spu, jnp, decimal=2)
    np.testing.assert_almost_equal(fn(0), 0.5, decimal=2)
    np.testing.assert_almost_equal(fn(100), 1, decimal=2)
    np.testing.assert_almost_equal(fn(-100), 0, decimal=2)


@pytest.mark.mpc
def test_t1(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.t1_sig)


@pytest.mark.mpc
def test_t3(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.t3_sig)


@pytest.mark.mpc
def test_t5(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.t5_sig)


@pytest.mark.mpc
def test_seg3(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.seg3_sig)


@pytest.mark.mpc
def test_df(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.df_sig)


@pytest.mark.mpc
def test_sr(sf_production_setup_devices):
    do_test(sf_production_setup_devices, sigmoid.sr_sig)
