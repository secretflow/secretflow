import jax
import jax.numpy as jnp
import numpy as np

import secretflow as sf
from tests.basecase import DeviceTestCase


class TestDevicePPUKernel(DeviceTestCase):
    def test_multiple_return(self):
        def foo(x, y):
            return x, y

        x, y = self.alice(np.random.rand)(3, 4), self.alice(np.random.rand)(3, 4)
        x_, y_ = x.to(self.ppu), y.to(self.ppu)
        x_, y_ = self.ppu(foo)(x_, y_)
        np.testing.assert_almost_equal(sf.reveal(x), sf.reveal(x_), decimal=6)

    def test_selu(self):
        def selu(x, alpha=1.67, lmbda=1.05):
            return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

        # PYU
        x = self.alice(np.random.rand)(3, 4)
        y = self.alice(selu)(x, 1.34)

        # PPU
        x_ = x.to(self.ppu)
        y_ = self.ppu(selu)(x_, 1.34).to(self.bob)

        np.testing.assert_almost_equal(sf.reveal(y), sf.reveal(y_), decimal=6)

    def test_mean(self):
        def get_weights():
            import tensorflow as tf

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
                tf.keras.layers.Dense(10, activation=tf.nn.relu),
                tf.keras.layers.Dense(3)
            ])

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            return model.get_weights()

        def average(*values, weights=None):
            return [jnp.average(values_zip, axis=0, weights=weights) for values_zip in zip(*values)]

        # PYU
        w1, w2 = self.alice(get_weights)(), self.alice(get_weights)()
        w = self.alice(average)(w1, w2.to(self.alice), weights=[1, 2])

        # PPU
        w1_, w2_ = w1.to(self.ppu), w2.to(self.ppu)
        w_ = self.ppu(average)(w1_, w2_, weights=[1, 2]).to(self.bob)

        for expected, actual in zip(sf.reveal(w), sf.reveal(w_)):
            np.testing.assert_almost_equal(expected, actual, decimal=6)

    def test_min(self):
        def min(*values):
            return jnp.min(jnp.stack(values), axis=0)

        # PYU
        m1, m2, m3 = self.alice(np.random.rand)(3, 4), self.bob(np.random.rand)(3, 4), self.carol(np.random.rand)(3, 4)
        m = self.alice(min)(m1, m2.to(self.alice), m3.to(self.alice))

        # PPU
        m1_, m2_, m3_ = m1.to(self.ppu), m2.to(self.ppu), m3.to(self.ppu)
        m_ = self.ppu(min)(m1_, m2_, m3_).to(self.bob)

        np.testing.assert_almost_equal(sf.reveal(m), sf.reveal(m_), decimal=6)

    def test_max(self):
        def max(*values):
            return jnp.min(jnp.stack(values), axis=0)

        # PYU
        m1, m2, m3 = self.alice(np.random.rand)(3, 4), self.bob(np.random.rand)(3, 4), self.carol(np.random.rand)(3, 4)
        m = self.alice(max)(m1, m2.to(self.alice), m3.to(self.alice))

        # PPU
        m1_, m2_, m3_ = m1.to(self.ppu), m2.to(self.ppu), m3.to(self.ppu)
        m_ = self.ppu(max)(m1_, m2_, m3_).to(self.bob)

        np.testing.assert_almost_equal(sf.reveal(m), sf.reveal(m_), decimal=6)

    def test_static_argument(self):
        def func(x, axis):
            return jnp.split(x, 2, axis)

        x = np.arange(10)
        x_ = sf.to(self.ppu, x)

        with self.assertRaises(jax.errors.ConcretizationTypeError):
            self.ppu(func)(x_, 0)

        y = func(x, 0)
        y_ = self.ppu(func, static_argnames='axis')(x_, axis=0)
        np.testing.assert_almost_equal(y, sf.reveal(y_), decimal=6)


class Test3PCPPUKernel(DeviceTestCase):
    def setUp(self) -> None:
        self.ppu = sf.PPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

    def test_matmul(self):
        # PYU
        x = self.alice(np.random.rand)(3, 4)
        y = self.bob(np.random.rand)(4, 5)

        # PPU
        z_ = self.ppu(lambda a, b: a @ b)(x.to(self.ppu), y.to(self.ppu))

        np.testing.assert_almost_equal(sf.reveal(x) @ sf.reveal(y), sf.reveal(z_), decimal=6)
