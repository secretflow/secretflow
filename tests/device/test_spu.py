import tempfile

import jax
import numpy as np
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, Relu

import secretflow as sf
from secretflow.device.device.spu import SPUObject
from tests.basecase import MultiDriverDeviceTestCase, SingleDriverDeviceTestCase


def MLP():
    nn_init, nn_apply = stax.serial(
        Dense(30),
        Relu,
        Dense(15),
        Relu,
        Dense(8),
        Relu,
        Dense(1),
    )

    return nn_init, nn_apply


def init_state(learning_rate):
    KEY = jax.random.PRNGKey(42)
    INPUT_SHAPE = (-1, 30)

    init_fun, predict_fun = MLP()
    _, params_init = init_fun(KEY, INPUT_SHAPE)
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_state = opt_init(params_init)
    return opt_state


class TestDeviceSPU(MultiDriverDeviceTestCase, SingleDriverDeviceTestCase):
    def test_scalar(self):
        x = self.alice(lambda: 1)()
        x_ = x.to(self.spu)
        self.assertEqual(x_.device, self.spu)
        y = x_.to(self.bob)
        np.testing.assert_almost_equal(sf.reveal(x), sf.reveal(y), decimal=5)

    def test_ndarray(self):
        x = self.alice(np.random.uniform)(-10, 10, (3, 4))
        x_ = x.to(self.spu)
        self.assertEqual(x_.device, self.spu)
        y = x_.to(self.bob)
        np.testing.assert_almost_equal(sf.reveal(x), sf.reveal(y), decimal=5)

    def test_pytree(self):
        x = self.alice(
            lambda: [
                [np.random.rand(3, 4), np.random.rand(4, 5)],
                {'weights': [1.0, 2.0]},
            ]
        )()
        x_ = x.to(self.spu)
        self.assertEqual(x_.device, self.spu)
        y = x_.to(self.bob)

        expected, actual = sf.reveal(x), sf.reveal(y)
        expected_flat, expected_tree = jax.tree_util.tree_flatten(expected)
        actual_flat, actual_tree = jax.tree_util.tree_flatten(actual)

        self.assertEqual(expected_tree, actual_tree)
        self.assertEqual(len(expected_flat), len(actual_flat))
        for expected, actual in zip(expected_flat, actual_flat):
            np.testing.assert_almost_equal(expected, actual, decimal=5)

    def test_to_heu(self):
        x = self.alice(np.random.uniform)(-10, 10, (3, 4))
        x_spu = x.to(self.spu)

        # spu -> heu
        x_heu = x_spu.to(self.heu)
        y = x_heu.to(self.alice)
        np.testing.assert_almost_equal(sf.reveal(x), sf.reveal(y), decimal=5)

        # heu -> spu
        x_spu = x_heu.to(self.spu)
        y = x_spu.to(self.alice)
        np.testing.assert_almost_equal(sf.reveal(x), sf.reveal(y), decimal=5)

    def test_dump_load(self):
        world_size = self.spu.world_size

        if world_size == 2:
            _, alice_path = tempfile.mkstemp()
            _, bob_path = tempfile.mkstemp()
            paths = [alice_path, bob_path]
        elif world_size == 3:
            _, alice_path = tempfile.mkstemp()
            _, bob_path = tempfile.mkstemp()
            _, carol_path = tempfile.mkstemp()
            paths = [alice_path, bob_path, carol_path]

        x = self.alice(np.random.uniform)(-10, 10, (3, 4))
        x_spu = x.to(self.spu)

        self.spu.dump(x_spu, paths)

        x_spu_ = self.spu.load(paths)
        assert isinstance(x_spu_, SPUObject)
        np.testing.assert_almost_equal(sf.reveal(x_spu), sf.reveal(x_spu_), decimal=5)
