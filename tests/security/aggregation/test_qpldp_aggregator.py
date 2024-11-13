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

import secretflow as sf
from secretflow.security.aggregation.experiment.qpldp_aggregator import QPLDPAggregator
from tests.security.aggregation.test_aggregator_base import AggregatorBase


@pytest.skip('Experimental, not work.', allow_module_level=True)
class TestQPLDPAggregator(AggregatorBase):
    @pytest.fixture()
    def env_and_aggregator(self, sf_production_setup_devices_ray):
        yield sf_production_setup_devices_ray, QPLDPAggregator(
            sf_production_setup_devices_ray.carol,
        )

    def test_average_on_list_with_weights_should_ok(
        test_average_on_list_with_weights_should_ok, env_and_aggregator
    ):
        env, aggregator = env_and_aggregator
        # GIVEN
        a = env.alice(
            lambda: [
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[21, 22, 23], [24, 25, 26]]),
            ]
        )()
        b = env.bob(
            lambda: [
                np.array([[11, 12, 13], [14, 15, 16]]),
                np.array([[31, 32, 33], [34, 35, 36]]),
            ]
        )()

        # WHEN
        avg = sf.reveal(aggregator.average([a, b], axis=0, weights=[2, 3]))

        # THEN
        np.testing.assert_almost_equal(
            avg[0], np.array([[7, 8, 9], [10, 11, 12]]), decimal=1
        )
        np.testing.assert_almost_equal(
            avg[1], np.array([[27, 28, 29], [30, 31, 32]]), decimal=1
        )

    def test_average_on_list_without_weights_should_ok(self, env_and_aggregator):
        env, aggregator = env_and_aggregator
        # GIVEN
        a = env.alice(
            lambda: [
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[21, 22, 23], [24, 25, 26]]),
            ]
        )()
        b = env.bob(
            lambda: [
                np.array([[11, 12, 13], [14, 15, 16]]),
                np.array([[31, 32, 33], [34, 35, 36]]),
            ]
        )()

        # WHEN
        avg = sf.reveal(aggregator.average([a, b], axis=0))

        # THEN
        np.testing.assert_almost_equal(
            avg[0], np.array([[6, 7, 8], [9, 10, 11]]), decimal=0
        )
        np.testing.assert_almost_equal(
            avg[1], np.array([[26, 27, 28], [29, 30, 31]]), decimal=0
        )
