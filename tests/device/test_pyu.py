import numpy as np

import secretflow.device as ft
from secretflow import reveal
from secretflow.device.device.pyu import PYUObject
from secretflow.device.device.spu import SPUObject
from tests.basecase import MultiDriverDeviceTestCase, SingleDriverDeviceTestCase


class TestDevicePYU(MultiDriverDeviceTestCase, SingleDriverDeviceTestCase):
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

    def test_dictionary_return(self):
        def load():
            return {'a': 1, 'b': 23}

        x = self.alice(load)()
        self.assertTrue(isinstance(x, ft.PYUObject))
        self.assertEqual(ft.reveal(x), {'a': 1, 'b': 23})

        x_ = x.to(self.spu)
        self.assertEqual(ft.reveal(x_), {'a': 1, 'b': 23})

    def test_to(self):
        @ft.with_device(self.alice)
        def load(*shape):
            return np.random.rand(*shape)

        x = load(3, 4)
        self.assertTrue(isinstance(x, PYUObject))

        x_1 = x.to(self.spu)
        self.assertTrue(isinstance(x_1, SPUObject))
        self.assertTrue(np.allclose(ft.reveal(x), ft.reveal(x_1)))
