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
