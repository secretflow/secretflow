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
import time

import numpy as np
import pytest

from secretflow.device.device.spu import SPUObject
from secretflow.device.driver import reveal, to, wait
from tests.sf_fixtures import SFProdParams


def _test_wait_should_ok(devices):
    # GIVEN
    def simple_write():
        _, temp_file = tempfile.mkstemp()
        time.sleep(5)
        with open(temp_file, 'w') as f:
            f.write('This is alice.')
        return temp_file

    o = devices.alice(simple_write)()

    # WHEN
    wait([o])

    # THEN
    def check(temp_file):
        with open(temp_file, 'r') as f:
            assert f.read() == 'This is alice.'
        return True

    file_path = reveal(o)
    assert reveal(devices.alice(check)(file_path))


_MPC_PARAMS_BRPC_RAY = {"cross_silo_comm_backend": "brpc_link", "ray_mode": True}


@pytest.mark.mpc(params=_MPC_PARAMS_BRPC_RAY)
def test_wait_should_ok_prod_brpc(sf_production_setup_devices):
    _test_wait_should_ok(sf_production_setup_devices)


def _test_spu_reveal(devices):
    with pytest.raises(ValueError):
        x = to(devices.spu, 32)

    x = to(devices.alice, 32).to(devices.spu)
    assert isinstance(x, SPUObject)

    x_ = reveal(x)
    assert isinstance(x_, np.ndarray)
    assert x_ == 32


@pytest.mark.mpc(params=_MPC_PARAMS_BRPC_RAY)
def test_spu_reveal_prod_brpc(sf_production_setup_devices):
    _test_spu_reveal(sf_production_setup_devices)


def _test_spu_reveal_empty_list(devices):
    x = to(devices.alice, []).to(devices.spu)
    assert isinstance(x, SPUObject)

    x_ = reveal(x)
    assert x_ == []


@pytest.mark.mpc(params=_MPC_PARAMS_BRPC_RAY)
def test_spu_reveal_empty_list_prod_brpc(sf_production_setup_devices):
    _test_spu_reveal_empty_list(sf_production_setup_devices)
