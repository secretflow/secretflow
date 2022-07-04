import numpy as np

import secretflow.device as ft
from secretflow import reveal
from tests.basecase import DeviceTestCase


class TestDevicePYU(DeviceTestCase):
    def test_device(self):
        @ft.with_device(self.alice)
        def load(*shape):
            return np.random.rand(*shape)

        x = load(3, 4)
        y = x.to(self.bob)
        self.assertEqual(x.device, self.alice)
        self.assertEqual(y.device, self.bob)
        np.testing.assert_equal(reveal(x), reveal(y))

    def test_average(self):
        def average(*a, axis=None, weights=None):
            return np.average(a, axis=axis, weights=weights)

        x = ft.with_device(self.alice)(np.random.rand)(3, 4)
        y = ft.with_device(self.bob)(np.random.rand)(3, 4)

        with self.assertRaises(AssertionError):
            self.alice(average)(x, y, axis=0)

        y = y.to(self.alice)
        actual = self.alice(average)(x, y, axis=0, weights=(1, 2))
        expected = np.average([reveal(x), reveal(y)], axis=0, weights=(1, 2))
        np.testing.assert_equal(reveal(actual), expected)

    def test_multiple_return(self):
        def load():
            return 1, 2, 3

        x, y, z = self.alice(load, num_returns=3)()
        self.assertTrue(isinstance(x, ft.PYUObject))

        x, y, z = ft.reveal([x, y, z])
        self.assertEqual(x, 1)
