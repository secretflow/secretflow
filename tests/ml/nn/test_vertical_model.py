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
from secretflow.ml.nn import SLModelTF
from secretflow.security.privacy import DPStrategy, GaussianEmbeddingDP, LabelDP
from tests.basecase import DeviceTestCase
from keras.datasets import mnist
from keras.utils import np_utils

_temp_dir = tempfile.mkdtemp()

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


def prepare_data(num_rows):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = x_train[:num_rows], y_train[:num_rows]
    x_train = x_train.astype('float32')
    x_train, x_test = x_train / 255, x_test / 255
    y_train = np_utils.to_categorical(y_train, 10)
    data_path = os.path.join(_temp_dir, f"sl_mnist.npz")
    np.savez(data_path, image=x_train, label=y_train)
    return data_path


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
        num_rows = 10000
        data_path = prepare_data(num_rows)
        fed_npz = load({self.alice: data_path, self.bob: data_path}, allow_pickle=True)

        data = fed_npz['image']
        label = fed_npz['label']

        alice_arr = self.alice(lambda: np.zeros(num_rows))()
        bob_arr = self.bob(lambda: np.zeros(num_rows))()
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
            num_samples=data.partition_shape()[self.alice][0],
            is_secure_generator=False,
        )
        dp_strategy_alice = DPStrategy(embedding_dp=gaussian_embedding_dp)
        label_dp = LabelDP(eps=64.0)
        dp_strategy_bob = DPStrategy(label_dp=label_dp)
        dp_strategy_dict = {self.alice: dp_strategy_alice, self.bob: dp_strategy_bob}
        dp_spent_step_freq = 10

        sl_model = SLModelTF(
            base_model_dict=base_model_dict,
            device_y=device_y,
            model_fuse=model_fuse,
            dp_strategy_dict=dp_strategy_dict,
        )
        history = sl_model.fit(
            data,
            label,
            validation_data=(data, label),
            epochs=1,
            batch_size=train_batch_size,
            shuffle=True,
            dp_spent_step_freq=dp_spent_step_freq,
        )
        global_metric = sl_model.evaluate(data, label, batch_size=128)

        zero_metric = sl_model.evaluate(
            data, label, sample_weight=sample_weights, batch_size=128
        )
        # test history
        self.assertAlmostEqual(
            global_metric['accuracy'], history['val_accuracy'][-1], 2
        )
        self.assertEquals(zero_metric['loss'], 0.0)
        self.assertGreater(global_metric['accuracy'], 0.88)
        result = sl_model.predict(data, batch_size=128, verbose=1)
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
        reload_metric = sl_model.evaluate(data, label, batch_size=128)
        self.assertAlmostEqual(global_metric['accuracy'], reload_metric['accuracy'], 1)
        self.assertAlmostEqual(global_metric['loss'], reload_metric['loss'], 1)
