# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocess
import pytest

import secretflow as sf
import secretflow.distributed as sfd
from secretflow.distributed.const import DISTRIBUTION_MODE
from tests.conftest import DeviceInventory


@pytest.fixture(scope="module")
def sf_tune_memory_setup_devices(request):
    devices = DeviceInventory()
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.DEBUG)
    sf.shutdown()
    sf.init(
        ["alice", "bob"],
        debug_mode=True,
    )

    devices.alice = sf.PYU("alice")
    devices.bob = sf.PYU("bob")

    yield devices
    del devices
    sf.shutdown()


@pytest.fixture(scope="module")
def sf_tune_simulation_setup_devices(request):
    devices = DeviceInventory()
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.SIMULATION)
    sf.shutdown()
    sf.init(
        ["alice", "bob"],
        address="local",
        num_cpus=32,
        log_to_driver=True,
        omp_num_threads=multiprocess.cpu_count(),
    )

    devices.alice = sf.PYU("alice")
    devices.bob = sf.PYU("bob")

    yield devices
    del devices
    sf.shutdown()
