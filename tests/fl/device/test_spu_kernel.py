# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax.numpy as jnp
import numpy as np
import pytest

import secretflow as sf


def _test_mean(devices):
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
    w1, w2 = devices.alice(get_weights)(), devices.alice(get_weights)()
    w = devices.alice(average)(w1, w2.to(devices.alice), weights=[1, 2])

    # SPU
    w1_, w2_ = w1.to(devices.spu), w2.to(devices.spu)
    w_ = devices.spu(average)(w1_, w2_, weights=[1, 2])

    for expected, actual in zip(sf.reveal(w), sf.reveal(w_)):
        np.testing.assert_almost_equal(expected, actual, decimal=5)


@pytest.mark.mpc
def test_mean_prod(sf_production_setup_devices):
    _test_mean(sf_production_setup_devices)


def test_mean_sim(sf_simulation_setup_devices):
    _test_mean(sf_simulation_setup_devices)
