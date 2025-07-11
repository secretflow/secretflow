# !/usr/bin/env python3
# *_* coding: utf-8 *_*
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

import tensorflow as tf

from secretflow_fl.ml.nn.fl.backend.tensorflow.strategy.fed_avg_g import FedAvgG

from ..model_def import mnist_conv_model


class TestFedAVG:
    def test_avg_g_local_step(self, sf_simulation_setup_devices):
        fed_avg_worker = FedAvgG(builder_base=mnist_conv_model)

        # prepare datset
        from tensorflow.keras.datasets import mnist  # type: ignore

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
