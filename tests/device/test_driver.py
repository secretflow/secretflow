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


@pytest.mark.mpc
def test_wait_should_ok_prod(sf_production_setup_devices):
    _test_wait_should_ok(sf_production_setup_devices)


def test_wait_should_ok_sim(sf_simulation_setup_devices):
    _test_wait_should_ok(sf_simulation_setup_devices)


def _test_wait_should_error_when_task_failure(devices):
    # GIVEN
    def task():
        raise AssertionError('This exception is expected by design.')

    o = devices.alice(task)()

    # WHEN
    with pytest.raises(AssertionError):
        wait([o])


# failing at this moment.
# def test_wait_should_error_when_task_failure_prod(sf_production_setup_devices):
#     _test_wait_should_error_when_task_failure(sf_production_setup_devices)


def test_wait_should_error_when_task_failure_sim(sf_simulation_setup_devices):
    _test_wait_should_error_when_task_failure(sf_simulation_setup_devices)


def _test_spu_reveal(devices):
    with pytest.raises(ValueError):
        x = to(devices.spu, 32)

    x = to(devices.alice, 32).to(devices.spu)
    assert isinstance(x, SPUObject)

    x_ = reveal(x)
    assert isinstance(x_, np.ndarray)
    assert x_ == 32


@pytest.mark.mpc
def test_spu_reveal_prod(sf_production_setup_devices):
    _test_spu_reveal(sf_production_setup_devices)


def test_spu_reveal_sim(sf_simulation_setup_devices):
    _test_spu_reveal(sf_simulation_setup_devices)


def _test_spu_reveal_empty_list(devices):
    x = to(devices.alice, []).to(devices.spu)
    assert isinstance(x, SPUObject)

    x_ = reveal(x)
    assert x_ == []


@pytest.mark.mpc
def test_spu_reveal_empty_list_prod(sf_production_setup_devices):
    _test_spu_reveal_empty_list(sf_production_setup_devices)


def test_spu_reveal_empty_list_sim(sf_simulation_setup_devices):
    _test_spu_reveal_empty_list(sf_simulation_setup_devices)
