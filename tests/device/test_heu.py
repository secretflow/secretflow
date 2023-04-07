import numpy as np
from heu import phe

import secretflow.device as ft
from secretflow import reveal
from secretflow.device.device.heu import HEUMoveConfig
from tests.basecase import MultiDriverDeviceTestCase, SingleDriverDeviceTestCase


class TestDeviceHEU(MultiDriverDeviceTestCase, SingleDriverDeviceTestCase):
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

        # Can't convert an HEUTensor to CPUTensor without secret key
        with self.assertRaises(AssertionError):
            y = x_.to(self.bob)

        y = x_.to(self.alice)
        self.assertEqual(y.device, self.alice)
        np.testing.assert_almost_equal(reveal(x), reveal(y), decimal=4)

    def test_math_ops(self):
        schema = phe.SchemaType.ZPaillier
        x = ft.with_device(self.alice)(np.random.rand)(3, 4)
        y = ft.with_device(self.bob)(np.random.rand)(3, 4)
        y_int = ft.with_device(self.bob)(np.random.randint)(10, size=(3, 4))
        z_int = ft.with_device(self.bob)(np.random.randint)(10, size=(4, 5))

        x_, y_, y_int_, z_int_ = (
            x.to(self.heu),  # x_ is ciphertext
            y.to(self.heu),
            y_int.to(
                self.heu, config=HEUMoveConfig(heu_encoder=phe.BigintEncoder(schema))
            ),
            z_int.to(
                self.heu, config=HEUMoveConfig(heu_encoder=phe.BigintEncoder(schema))
            ),
        )  # plaintext

        add_ = x_ + y_  # shape: 3x4
        sub_ = x_ - y_  # shape: 3x4
        # x_ is 1x scaled, y_int_ is not scaled, so result is 1x scaled
        mul_ = x_ * y_int_  # shape: 3x4
        matmul_ = x_ @ z_int_  # shape: 3x5

        x, y, y_int, z_int = reveal(x), reveal(y), reveal(y_int), reveal(z_int)
        add, sub = add_.to(self.alice), sub_.to(self.alice)
        mul = mul_.to(self.alice)
        matmul = matmul_.to(self.alice)

        np.testing.assert_almost_equal(x + y, reveal(add), decimal=4)
        np.testing.assert_almost_equal(x - y, reveal(sub), decimal=4)
        np.testing.assert_almost_equal(x * y_int, reveal(mul), decimal=4)
        np.testing.assert_almost_equal(x @ z_int, reveal(matmul), decimal=4)

        # test slice
        add = reveal(add_[2, 3].to(self.alice))
        sub = reveal(sub_[1:3, :].to(self.alice))
        mul = reveal(mul_[:3:2, ::-1].to(self.alice))
        matmul = reveal(matmul_[[0, 1, 2], 1::2].to(self.alice))

        self.assertTrue(isinstance(add, float))  # add is scalar
        self.assertEqual(sub.shape, (2, 4))
        self.assertEqual(mul.shape, (2, 4))
        self.assertEqual(matmul.shape, (3, 2))
        np.testing.assert_almost_equal((x + y)[2, 3], add, decimal=4)
        np.testing.assert_almost_equal((x - y)[1:3, :], sub, decimal=4)
        np.testing.assert_almost_equal((x * y_int)[:3:2, ::-1], mul, decimal=4)
        np.testing.assert_almost_equal((x @ z_int)[[0, 1, 2], 1::2], matmul, decimal=4)

    def test_sum(self):
        # test vector, ciphertext
        m = ft.with_device(self.alice)(np.random.rand)(20)
        m_heu = m.to(self.heu)  # ciphertext
        np.testing.assert_almost_equal(reveal(m).sum(), reveal(m_heu.sum()), decimal=4)
        np.testing.assert_almost_equal(
            reveal(m)[[1, 2, 3]].sum(), reveal(m_heu[[1, 2, 3]].sum()), decimal=4
        )
        np.testing.assert_almost_equal(
            reveal(m)[3:10].sum(), reveal(m_heu[3:10].sum()), decimal=4
        )

        # test matrix
        m = ft.with_device(self.bob)(np.random.rand)(20, 20)
        m_heu = m.to(
            self.heu, HEUMoveConfig(heu_dest_party=self.bob.party)
        )  # plaintext
        self.assertTrue(m_heu.is_plain)
        np.testing.assert_almost_equal(reveal(m).sum(), reveal(m_heu.sum()), decimal=4)
        np.testing.assert_almost_equal(
            reveal(m).sum(), reveal(m_heu.encrypt().sum()), decimal=4
        )
