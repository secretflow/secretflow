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

import secretflow as sf


def _test_reveal(devices):
    x = devices.alice(lambda: np.random.rand(3, 4))()
    vals = sf.reveal(
        {
            devices.spu: x.to(
                devices.spu,
            ),
            devices.carol: x.to(devices.carol),
        }
    )
    x = sf.reveal(x)
    np.testing.assert_almost_equal(x, vals[devices.spu], decimal=1)
    np.testing.assert_almost_equal(x, vals[devices.carol], decimal=1)


def test_reveal_multiple_driver(sf_production_setup_devices):
    _test_reveal(sf_production_setup_devices)


def test_reveal_single_driver(sf_simulation_setup_devices):
    _test_reveal(sf_simulation_setup_devices)
