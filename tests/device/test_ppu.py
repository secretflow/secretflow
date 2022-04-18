import jax
import numpy as np

import secretflow.device as ft
from tests.basecase import DeviceTestCase, cluster_def
from secretflow import reveal


class TestDevicePPU(DeviceTestCase):
    def test_scalar(self):
        x = self.alice(lambda: 1)()
        x_ = x.to(self.ppu)
        self.assertEqual(x_.device, self.ppu)
        y = x_.to(self.bob)
        np.testing.assert_almost_equal(reveal(x), reveal(y), decimal=6)

    def test_ndarray(self):
        x = self.alice(np.random.uniform)(-10, 10, (3, 4))
        x_ = x.to(self.ppu)
        self.assertEqual(x_.device, self.ppu)
        y = x_.to(self.bob)
        np.testing.assert_almost_equal(reveal(x), reveal(y), decimal=6)

    def test_pytree(self):
        x = self.alice(lambda: [[np.random.rand(3, 4), np.random.rand(4, 5)], {'weights': [1.0, 2.0]}])()
        x_ = x.to(self.ppu)
        self.assertEqual(x_.device, self.ppu)
        y = x_.to(self.bob)

        expected, actual = reveal(x), reveal(y)
        expected_flat, expected_tree = jax.tree_util.tree_flatten(expected)
        actual_flat, actual_tree = jax.tree_util.tree_flatten(actual)

        self.assertEqual(expected_tree, actual_tree)
        self.assertEqual(len(expected_flat), len(actual_flat))
        for expected, actual in zip(expected_flat, actual_flat):
            np.testing.assert_almost_equal(expected, actual, decimal=6)

    def test_to_heu(self):
        x = self.alice(np.random.uniform)(-10, 10, (3, 4))
        x_ppu = x.to(self.ppu)
        x_heu = x_ppu.to(self.heu)
        y = x_heu.to(self.alice)

        scale = 1 << ft.fxp_precision(cluster_def['runtime_config']['field'])
        np.testing.assert_almost_equal(reveal(x), reveal(y) / scale, decimal=6)
