#!/usr/bin/env python3
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

import numpy as np
import tensorflow as tf

from secretflow.data import FedNdarray, PartitionWay
from secretflow.device import reveal
from secretflow_fl.ml.nn import SLModel

# Prepare dataset
np.random.seed(7)
n, d = 1024, 8
train_feature = np.random.uniform(0, 1, [n, d])
valid_feature = np.random.uniform(0, 1, [n, d])
coef = np.random.uniform(0, 1, [d])
train_label = (train_feature @ coef > 0.5).astype(float)
valid_label = (valid_feature @ coef > 0.5).astype(float)


class SplitMLP(tf.keras.layers.Layer):
    def __init__(self):
        super(SplitMLP, self).__init__()
        np.random.seed(7)
        k_np = np.random.uniform(0, 1, [4, 1])
        init = tf.keras.initializers.Constant(k_np)

        self.basenet_alice = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    1,
                    activation=None,
                    kernel_initializer=init,
                    use_bias=False,
                ),
            ]
        )
        self.basenet_bob = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    1,
                    activation=None,
                    kernel_initializer=init,
                    use_bias=False,
                ),
            ]
        )

    # fuse net
    def call(self, alice_feature, bob_feature):
        h_alice = self.basenet_alice(alice_feature)
        h_bob = self.basenet_bob(bob_feature)
        h = h_alice + h_bob
        return h


def create_base_model(name="base_model"):
    # Create model
    def create_model():
        import tensorflow as tf
        from tensorflow import keras

        np.random.seed(7)
        k_np = np.random.uniform(0, 1, [4, 1])
        init = tf.keras.initializers.Constant(k_np)

        model = keras.Sequential(
            [
                keras.Input(shape=4),
                keras.layers.Dense(
                    1,
                    activation=None,
                    kernel_initializer=init,
                    use_bias=False,
                ),
            ]
        )

        # Compile model
        model.summary()
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=[tf.keras.metrics.AUC()],
        )
        return model

    return create_model


def create_fuse_model(name="fuse_model"):
    def create_model():
        import tensorflow as tf
        from tensorflow import keras

        #  input
        input_layers = [
            keras.Input(
                1,
            ),
            keras.Input(
                1,
            ),
        ]
        output = tf.nn.sigmoid(input_layers[0] + input_layers[1])
        model = keras.Model(inputs=input_layers, outputs=output)

        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=[tf.keras.metrics.AUC()],
        )
        return model

    return create_model


def build_tf_model():
    alice_feature = tf.keras.Input(shape=(4,), name="alice_feature")
    bob_feature = tf.keras.Input(shape=(4,), name="bob_feature")
    split_mlp = SplitMLP()
    logits = split_mlp(alice_feature, bob_feature)
    model = tf.keras.Model(
        inputs=[alice_feature, bob_feature], outputs=tf.sigmoid(logits)
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=[tf.keras.metrics.AUC()],
    )
    return model


class TestSLModelTFCorrectness:
    def test_sl_model_correctness_with_tf(self, sf_simulation_setup_devices):
        train_x = FedNdarray(
            partitions={
                sf_simulation_setup_devices.alice: sf_simulation_setup_devices.alice(
                    lambda: train_feature[:, :4]
                )(),
                sf_simulation_setup_devices.bob: sf_simulation_setup_devices.bob(
                    lambda: train_feature[:, 4:]
                )(),
            },
            partition_way=PartitionWay.VERTICAL,
        )

        train_y = FedNdarray(
            partitions={
                sf_simulation_setup_devices.alice: sf_simulation_setup_devices.alice(
                    lambda: train_label
                )(),
            },
            partition_way=PartitionWay.VERTICAL,
        )

        valid_x = FedNdarray(
            partitions={
                sf_simulation_setup_devices.alice: sf_simulation_setup_devices.alice(
                    lambda: train_feature[:, :4]
                )(),
                sf_simulation_setup_devices.bob: sf_simulation_setup_devices.bob(
                    lambda: train_feature[:, 4:]
                )(),
            },
            partition_way=PartitionWay.VERTICAL,
        )

        valid_y = FedNdarray(
            partitions={
                sf_simulation_setup_devices.alice: sf_simulation_setup_devices.alice(
                    lambda: train_label
                )(),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        base_model_dict = {
            sf_simulation_setup_devices.alice: create_base_model(),
            sf_simulation_setup_devices.bob: create_base_model(),
        }
        model_fuse = create_fuse_model()

        sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=sf_simulation_setup_devices.alice,
            model_fuse=model_fuse,
            random_seed=7,
        )

        # hyper paremeter
        batch_size = 128
        epochs = 1

        sl_history = sl_model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=1,
            validation_freq=1,
            random_seed=7,
        )

        # TF part
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "alice_feature": train_feature[:, :4],
                    "bob_feature": train_feature[:, 4:],
                },
                train_label,
            )
        ).batch(batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "alice_feature": valid_feature[:, :4],
                    "bob_feature": valid_feature[:, 4:],
                },
                valid_label,
            )
        ).batch(batch_size)

        tf_model = build_tf_model()
        tf_history = tf_model.fit(
            train_dataset, epochs=epochs, validation_data=valid_dataset
        )
        assert sl_history["val_auc_1"][-1] == tf_history.history["val_auc"][-1]

        sl_model_alice = reveal(
            sl_model._workers[sf_simulation_setup_devices.alice].get_base_weights()
        )
        sl_model_bob = reveal(
            sl_model._workers[sf_simulation_setup_devices.bob].get_base_weights()
        )
        tf_origin_model = tf_model.get_weights()

        assert (sl_model_alice[0] == tf_origin_model[0]).all()
        assert (sl_model_bob[0] == tf_origin_model[1]).all()
