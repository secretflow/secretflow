from typing import Tuple

import jax.numpy as jnp
import numpy as np

import secretflow as sf
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy
from tests.basecase import DeviceTestCase


class TestDeviceSPUKernel(DeviceTestCase):
    def test_multiple_return(self):
        def foo(x, y) -> Tuple[int, int]:
            return x, y

        x, y = self.alice(np.random.rand)(3, 4), self.alice(np.random.rand)(3, 4)
        x_, y_ = x.to(self.spu), y.to(self.spu)

        z_ = self.spu(foo)(x_, y_)

        np.testing.assert_almost_equal(
            (sf.reveal(x), sf.reveal(y)), sf.reveal(z_), decimal=6
        )

        x_hat, y_hat = self.spu(
            foo, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_COMPILER
        )(x_, y_)
        np.testing.assert_almost_equal(
            (sf.reveal(x), sf.reveal(y)),
            (sf.reveal(x_hat), sf.reveal(y_hat)),
            decimal=6,
        )

        x_hat_2, y_hat_2 = self.spu(
            foo,
            num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
            user_specified_num_returns=2,
        )(x_, y_)
        np.testing.assert_almost_equal(
            (sf.reveal(x), sf.reveal(y)),
            (sf.reveal(x_hat_2), sf.reveal(y_hat_2)),
            decimal=6,
        )

    def test_selu(self):
        def selu(x, alpha=1.67, lmbda=1.05):
            return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

        # PYU
        x = self.alice(np.random.rand)(3, 4)
        y = self.alice(selu)(x, 1.34)

        # SPU
        x_ = x.to(self.spu)
        y_ = self.spu(selu)(x_, 1.34).to(self.bob)

        np.testing.assert_almost_equal(sf.reveal(y), sf.reveal(y_), decimal=6)

    def test_mean(self):
        def get_weights():
            import tensorflow as tf

            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        10, activation=tf.nn.relu, input_shape=(4,)
                    ),  # input shape required
                    tf.keras.layers.Dense(10, activation=tf.nn.relu),
                    tf.keras.layers.Dense(3),
                ]
            )

            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
            )

            return model.get_weights()

        def average(*values, weights=None):
            return [
                jnp.average(
                    jnp.array(values_zip),
                    axis=0,
                    weights=jnp.array(weights) if weights else None,
                )
                for values_zip in zip(*values)
            ]

        # PYU
        w1, w2 = self.alice(get_weights)(), self.alice(get_weights)()
        w = self.alice(average)(w1, w2.to(self.alice), weights=[1, 2])

        # SPU
        w1_, w2_ = w1.to(self.spu), w2.to(self.spu)
        w_ = self.spu(average)(w1_, w2_, weights=[1, 2])

        for expected, actual in zip(sf.reveal(w), sf.reveal(w_)):
            np.testing.assert_almost_equal(expected, actual, decimal=5)

    def test_min(self):
        def min(*values):
            return jnp.min(jnp.stack(values), axis=0)

        # PYU
        m1, m2, m3 = (
            self.alice(np.random.rand)(3, 4),
            self.bob(np.random.rand)(3, 4),
            self.carol(np.random.rand)(3, 4),
        )
        m = self.alice(min)(m1, m2.to(self.alice), m3.to(self.alice))

        # SPU
        m1_, m2_, m3_ = m1.to(self.spu), m2.to(self.spu), m3.to(self.spu)
        m_ = self.spu(min)(m1_, m2_, m3_).to(self.bob)

        np.testing.assert_almost_equal(sf.reveal(m), sf.reveal(m_), decimal=6)

    def test_max(self):
        def max(*values):
            return jnp.min(jnp.stack(values), axis=0)

        # PYU
        m1, m2, m3 = (
            self.alice(np.random.rand)(3, 4),
            self.bob(np.random.rand)(3, 4),
            self.carol(np.random.rand)(3, 4),
        )
        m = self.alice(max)(m1, m2.to(self.alice), m3.to(self.alice))

        # SPU
        m1_, m2_, m3_ = m1.to(self.spu), m2.to(self.spu), m3.to(self.spu)
        m_ = self.spu(max)(m1_, m2_, m3_).to(self.bob)

        np.testing.assert_almost_equal(sf.reveal(m), sf.reveal(m_), decimal=6)

    def test_static_argument(self):
        def func(x, axis):
            return jnp.split(x, 2, axis)

        x = np.arange(10)
        x_ = sf.to(self.spu, x)

        # with self.assertRaises(jax.errors.ConcretizationTypeError):
        #    self.spu(func)(x_, 0)

        y = func(x, 0)
        y_ = self.spu(func, static_argnames='axis')(x_, axis=0)
        np.testing.assert_almost_equal(y, sf.reveal(y_), decimal=6)

        def init_w(base: float, num_feat: int) -> np.ndarray:
            # last one is bias
            return jnp.full((num_feat + 1, 1), base, dtype=jnp.float32)

        spu_w = self.spu(init_w, static_argnames=('base', 'num_feat'))(
            base=0, num_feat=30
        )
        print(sf.reveal(spu_w))


class Test3PCSPUKernel(DeviceTestCase):
    def setUp(self) -> None:
        self.spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

    def test_matmul(self):
        # PYU
        x = self.alice(np.random.rand)(3, 4)
        y = self.bob(np.random.rand)(4, 5)

        # SPU
        z_ = self.spu(lambda a, b: a @ b)(x.to(self.spu), y.to(self.spu))

        np.testing.assert_almost_equal(
            sf.reveal(x) @ sf.reveal(y), sf.reveal(z_), decimal=6
        )
