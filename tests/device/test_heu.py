import numpy as np

import secretflow.device as ft
from secretflow import reveal
from tests.basecase import DeviceTestCase


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
        np.testing.assert_almost_equal(reveal(x), reveal(y), decimal=4)

    def test_math_ops(self):
        x = ft.with_device(self.alice)(np.random.rand)(3, 4)
        y = ft.with_device(self.bob)(np.random.rand)(3, 4)
        y_int = ft.with_device(self.bob)(np.random.randint)(10, size=(3, 4))
        z_int = ft.with_device(self.bob)(np.random.randint)(10, size=(4, 5))

        x_ = x.to(self.heu)  # x_ is ciphertext
        y_, y_int_, z_int_ = (
            y.to(self.heu),
            y_int.to(self.heu),
            z_int.to(self.heu),
        )  # plaintext

        add_ = x_ + y_  # shape: 3x4
        sub_ = x_ - y_  # shape: 3x4
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

        self.assertEqual(add.shape, ())  # scalar
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

        def test_matrix(m, m_heu):
            np.testing.assert_almost_equal(
                reveal(m).sum(), reveal(m_heu.sum()), decimal=4
            )
            np.testing.assert_almost_equal(
                reveal(m).sum(axis=0), reveal(m_heu.sum(axis=0)), decimal=4
            )
            np.testing.assert_almost_equal(
                reveal(m).sum(axis=-1), reveal(m_heu.sum(axis=-1)), decimal=4
            )
            np.testing.assert_almost_equal(
                reveal(m).sum(axis=(0, 1)), reveal(m_heu.sum(axis=(0, 1))), decimal=4
            )

        # test matrix, plaintext
        m = ft.with_device(self.bob)(np.random.rand)(20, 20)
        m_heu = m.to(self.heu, heu_dest_party=self.bob.party)  # plaintext
        test_matrix(m, m_heu)
        test_matrix(m, m_heu.encrypt())

        # test 3d-tensor from bob
        m = ft.with_device(self.bob)(np.random.rand)(5, 10, 15)
        m_heu = m.to(self.heu, heu_dest_party=self.bob.party)  # plaintext
        test_matrix(m, m_heu)
        test_matrix(m, m_heu.encrypt())

        # test 4d-tensor from alice
        m = ft.with_device(self.alice)(np.random.rand)(13, 11, 7, 5)
        m_heu = m.to(self.heu, heu_dest_party=self.bob.party)  # ciphertext
        test_matrix(m, m_heu)
