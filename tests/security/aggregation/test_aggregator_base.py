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


# NOTE(fengjun.feng): could not use TestAggregatorBase since pytest would recognize as a testsuite.
class AggregatorBase:
    def test_sum_on_single_should_ok(self, env_and_aggregator):
        env, aggregator = env_and_aggregator
        # GIVEN
        a = env.alice(lambda: np.array([[1.0, 2.0, 3], [4.0, 5.0, 6.0]]))()
        b = env.bob(lambda: np.array([[11.0, 12.0, 13.0], [14, 15.0, 16.0]]))()

        # WHEN
        sum = sf.reveal(aggregator.sum([a, b], axis=0))

        # THEN
        np.testing.assert_almost_equal(
            sum, np.array([[12.0, 14.0, 16.0], [18.0, 20.0, 22.0]]), decimal=5
        )

    def test_sum_on_list_should_ok(self, env_and_aggregator):
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
        sum = sf.reveal(aggregator.sum([a, b], axis=0))

        # THEN
        np.testing.assert_almost_equal(
            sum[0], np.array([[12, 14, 16], [18, 20, 22]]), decimal=5
        )
        np.testing.assert_almost_equal(
            sum[1], np.array([[52, 54, 56], [58, 60, 62]]), decimal=5
        )

    def test_average_on_single_without_weights_should_ok(self, env_and_aggregator):
        env, aggregator = env_and_aggregator
        # GIVEN
        a = env.alice(lambda: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))()
        b = env.bob(lambda: np.array([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]))()

        # WHEN
        avg = sf.reveal(aggregator.average([a, b], axis=0))

        # THEN
        np.testing.assert_almost_equal(
            avg, np.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]), decimal=5
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
            avg[0], np.array([[6, 7, 8], [9, 10, 11]]), decimal=5
        )
        np.testing.assert_almost_equal(
            avg[1], np.array([[26, 27, 28], [29, 30, 31]]), decimal=5
        )

    def test_average_with_weights_should_ok(self, env_and_aggregator):
        env, aggregator = env_and_aggregator
        # GIVEN
        a = env.alice(lambda: np.array([[1, 2, 3], [4, 5, 6]]))()
        b = env.bob(lambda: np.array([[11, 12, 13], [14, 15, 16]]))()

        # WHEN
        sum = sf.reveal(aggregator.average([a, b], axis=0, weights=[2, 3]))

        # THEN
        np.testing.assert_almost_equal(
            sum, np.array([[7, 8, 9], [10, 11, 12]]), decimal=4
        )

    def test_average_on_list_with_weights_should_ok(self, env_and_aggregator):
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
            avg[0], np.array([[7, 8, 9], [10, 11, 12]]), decimal=4
        )
        np.testing.assert_almost_equal(
            avg[1], np.array([[27, 28, 29], [30, 31, 32]]), decimal=4
        )

    def test_average_with_same_shape_weights_should_ok(self, env_and_aggregator):
        env, aggregator = env_and_aggregator
        # GIVEN
        arr0 = np.array([[1, 2, 3]])
        arr1 = np.array([[11, 12, 13]])
        a = env.alice(lambda: arr0)()
        b = env.bob(lambda: arr1)()

        weights = np.array([[[5, 7, 2]], [[5, 3, 8]]])

        # WHEN
        sum = sf.reveal(aggregator.average([a, b], axis=0, weights=weights))

        # THEN
        np.testing.assert_almost_equal(
            sum, np.average([arr0, arr1], axis=0, weights=weights), decimal=4
        )
