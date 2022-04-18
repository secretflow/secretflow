import numpy as np

import secretflow.device as ft
from tests.basecase import DeviceTestCase
from secretflow import reveal


class TestDeviceHEU(DeviceTestCase):
    def test_device(self):
        x = ft.with_device(self.alice)(np.random.rand)(3, 4)
        x_ = x.to(self.heu)
        self.assertEqual(x_.device, self.heu)

        # Can't place a user-defined function to HEU Device
        with self.assertRaises(NotImplementedError):
            @ft.with_device(self.heu)
            def add(a, b):
                return a + b

            y = add(x_, x_)

        # Can't convert a HEUTensor to CPUTensor without secret key
        with self.assertRaises(AssertionError):
            y = x_.to(self.bob)

        y = x_.to(self.alice)
        self.assertEqual(y.device, self.alice)
        np.testing.assert_equal(reveal(x), reveal(y))

    def test_math_ops(self):
        x = ft.with_device(self.alice)(np.random.rand)(3, 4)
        y = ft.with_device(self.bob)(np.random.rand)(3, 4)
        z = ft.with_device(self.bob)(np.random.rand)(4, 5)

        x_, y_, z_ = x.to(self.heu), y.to(self.heu), z.to(self.heu)

        add_ = x_ + y_
        sub_ = x_ - y_
        mul_ = x_ * y_
        matmul_ = x_ @ z_

        x, y, z = reveal(x), reveal(y), reveal(z)
        add, sub, mul, matmul = add_.to(self.alice), sub_.to(self.alice), mul_.to(self.alice), matmul_.to(self.alice)

        np.testing.assert_almost_equal(x + y, reveal(add))
        np.testing.assert_almost_equal(x - y, reveal(sub))
        np.testing.assert_almost_equal(x * y, reveal(mul))
        np.testing.assert_almost_equal(x @ z, reveal(matmul))
