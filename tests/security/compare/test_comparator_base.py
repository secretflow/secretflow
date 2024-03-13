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

from secretflow import reveal


# NOTE(fengjun.feng): could not use TestComparatorBase since pytest would recognize as a testsuite.
class ComparatorBase:
    def test_min_should_ok(self, env_and_comparator):
        env, comparator = env_and_comparator
        # GIVEN
        a = env.alice(lambda: np.array([[1, 2, 3], [14, 15, 16]]))()
        b = env.bob(lambda: np.array([[11, 12, 13], [4, 5, 6]]))()

        # WHEN
        min = reveal(comparator.min([a, b], axis=0))

        # THEN
        np.testing.assert_equal(reveal(min), np.array([[1, 2, 3], [4, 5, 6]]))

    def test_max_should_ok(self, env_and_comparator):
        env, comparator = env_and_comparator
        # GIVEN
        a = env.alice(lambda: np.array([[1, 2, 3], [14, 15, 16]]))()
        b = env.bob(lambda: np.array([[11, 12, 13], [4, 5, 6]]))()

        # WHEN
        max = reveal(comparator.max([a, b], axis=0))

        # THEN
        np.testing.assert_equal(reveal(max), np.array([[11, 12, 13], [14, 15, 16]]))
