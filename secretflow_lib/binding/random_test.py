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

import unittest

import numpy as np

import secretflow_lib.binding._lib.random as random


class TestRandom(unittest.TestCase):
    def test_uniform_real(self):
        x = random.uniform_real(0, 1, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))

    def test_bernoulli_neg_exp(self):
        x = random.bernoulli_neg_exp(0.5, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))

    def test_normal_real(self):
        x = random.secure_normal_real(0, 1, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))

    def test_normal_discrete(self):
        x = random.normal_discrete(0, 1, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(x.dtype, np.int32)

    def test_laplace_real(self):
        x = random.secure_laplace_real(0, 1, size=(3, 4))
        self.assertEqual(x.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
