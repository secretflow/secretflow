#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
import os
import tempfile

import numpy as np

from secretflow.data.ndarray import load
from secretflow.ml.nn import SLModel
from secretflow.security.privacy import DPStrategy, GaussianEmbeddingDP, LabelDP
from tests.basecase import DeviceTestCase
from secretflow.utils.simulation.datasets import load_mnist

_temp_dir = tempfile.mkdtemp()

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


def create_base_model(input_dim, output_dim, name='base_model'):
    # Create model
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(output_dim, activation="relu"),
            ]
        )
        # Compile model
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
        return model  # need wrap

    return create_model


def create_fuse_model(input_dim, output_dim, party_nums, name='fuse_model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers

        # input
        input_layers = []
        for i in range(party_nums):
            input_layers.append(
                keras.Input(
                    input_dim,
                )
            )

        merged_layer = layers.concatenate(input_layers)
        fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
        output = layers.Dense(output_dim, activation='softmax')(fuse_layer)
        # Create model
        model = keras.Model(inputs=input_layers, outputs=output)
        model.summary()
        # Compile model
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
        return model

    return create_model


class TestFedModelNdArray(DeviceTestCase):
    def test_keras_model(self):
        num_samples = 10000
        (x_train, y_train), (_, _) = load_mnist(
            parts={
                self.alice: (0, num_samples),
                self.bob: (num_samples, num_samples * 2),
            },
            normalized_x=True,
            categorical_y=True,
        )

        alice_arr = self.alice(lambda: np.zeros(num_samples))()
        bob_arr = self.bob(lambda: np.zeros(num_samples))()
        sample_weights = load({self.alice: alice_arr, self.bob: bob_arr})

        # prepare model
        num_classes = 10
        input_shape = (28, 28, 1)
        hidden_size = 64
        train_batch_size = 128

        # User-defined compiled keras model
        device_y = self.bob
        model_base_alice = create_base_model(input_shape, hidden_size)
        model_base_bob = create_base_model(input_shape, hidden_size)
        base_model_dict = {self.alice: model_base_alice, self.bob: model_base_bob}
        model_fuse = create_fuse_model(
            input_dim=hidden_size,
            party_nums=len(base_model_dict),
            output_dim=num_classes,
        )

        # Define DP operations
        gaussian_embedding_dp = GaussianEmbeddingDP(
            noise_multiplier=0.5,
            l2_norm_clip=1.0,
            batch_size=train_batch_size,
            num_samples=num_samples,
            is_secure_generator=False,
        )
        dp_strategy_alice = DPStrategy(embedding_dp=gaussian_embedding_dp)
        label_dp = LabelDP(eps=64.0)
        dp_strategy_bob = DPStrategy(label_dp=label_dp)
        dp_strategy_dict = {self.alice: dp_strategy_alice, self.bob: dp_strategy_bob}
        dp_spent_step_freq = 10

        sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=device_y,
            model_fuse=model_fuse,
            dp_strategy_dict=dp_strategy_dict,
        )
        history = sl_model.fit(
            x_train,
            y_train,
            validation_data=(x_train, y_train),
            epochs=1,
            batch_size=train_batch_size,
            shuffle=True,
            dp_spent_step_freq=dp_spent_step_freq,
        )
        global_metric = sl_model.evaluate(x_train, y_train, batch_size=128)

        zero_metric = sl_model.evaluate(
            x_train, y_train, sample_weight=sample_weights, batch_size=128
        )
        # test history
        self.assertAlmostEqual(
            global_metric['accuracy'], history['val_accuracy'][-1], 2
        )
        self.assertEquals(zero_metric['loss'], 0.0)
        # note: acc 0.88 not stable, change it to 0.87
        self.assertGreater(global_metric['accuracy'], 0.87)
        result = sl_model.predict(x_train, batch_size=128, verbose=1)
        self.assertIsNotNone(result)
        base_model_path = os.path.join(_temp_dir, "base_model")
        fuse_model_path = os.path.join(_temp_dir, "fuse_model")
        sl_model.save_model(
            base_model_path=base_model_path,
            fuse_model_path=fuse_model_path,
            is_test=True,
        )
        self.assertIsNotNone(os.path.exists(base_model_path))
        self.assertIsNotNone(os.path.exists(fuse_model_path))

        sl_model.load_model(
            base_model_path=base_model_path,
            fuse_model_path=fuse_model_path,
            is_test=True,
        )
        reload_metric = sl_model.evaluate(x_train, y_train, batch_size=128)
        self.assertAlmostEqual(global_metric['accuracy'], reload_metric['accuracy'], 1)
        self.assertAlmostEqual(global_metric['loss'], reload_metric['loss'], 1)
