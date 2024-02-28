# !/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""

import tensorflow as tf

from secretflow.ml.nn.fl.backend.tensorflow.strategy.fed_avg_g import FedAvgG
from tests.ml.nn.fl.model_def import mnist_conv_model


class TestFedAVG:
    def test_avg_g_local_step(self, sf_simulation_setup_devices):
        fed_avg_worker = FedAvgG(builder_base=mnist_conv_model)

        # prepare datset
        from tensorflow.keras.datasets import mnist

        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test / 255.0
        y_test = tf.one_hot(y_test, depth=10)
        fed_avg_worker.train_set = iter(
            tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
        )
        gradients = None
        gradients, num_sample = fed_avg_worker.train_step(
            gradients, cur_steps=0, train_steps=1
        )
        fed_avg_worker.apply_weights(gradients)

        assert num_sample == 128

        assert len(gradients) == 6
        _, num_sample = fed_avg_worker.train_step(gradients, cur_steps=1, train_steps=2)
        assert num_sample == 256
