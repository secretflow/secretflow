#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
import tempfile
import os

import numpy as np

from tests.basecase import DeviceTestCase
from secretflow.data.ndarray import load
from secretflow.model.sl_model import SLModelTF

_temp_dir = tempfile.mkdtemp()

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


def create_base_model(input_dim, output_dim,  name='base_model'):
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
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy"])
        return model  # 不能序列化的
    return create_model


def create_fuse_model(input_dim, output_dim, party_nums, name='fuse_model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        # input
        input_layers = []
        for i in range(party_nums):
            input_layers.append(keras.Input(input_dim,))

        # TODO: 是否应该把融合过程剥离出device_y，只给label方融合后的hidden，目前这样是有安全问题的
        merged_layer = layers.concatenate(input_layers)
        fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
        output = layers.Dense(output_dim, activation='softmax')(fuse_layer)
        # Create model
        model = keras.Model(inputs=input_layers, outputs=output)
        model.summary()
        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy"])
        return model
    return create_model


class TestFedModelNdArray(DeviceTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.path_x = 'tests/datasets/mnist/vertical/mnist_x_mini.npy'
        cls.path_y = 'tests/datasets/mnist/vertical/mnist_y_mini.npy'

    def test_keras_model(self):

        data = load({self.alice: self.path_x,
                     self.bob: self.path_x}, allow_pickle=True)
        label = load({self.alice: self.path_y,
                      self.bob: self.path_y}, allow_pickle=True)

        alice_arr = self.alice(lambda: np.zeros(10000))()
        bob_arr = self.bob(lambda: np.zeros(10000))()
        sample_weights = load({self.alice: alice_arr, self.bob: bob_arr})

        # prepare model
        num_classes = 10

        input_shape = (28, 28, 1)
        hidden_size = 64

        # 用户定义的已编译后的keras model
        device_y = self.bob
        model_base_alice = create_base_model(input_shape, hidden_size)
        model_base_bob = create_base_model(input_shape, hidden_size)
        base_model_dict = {
            self.alice: model_base_alice,
            self.bob:   model_base_bob
        }
        model_fuse = create_fuse_model(
            input_dim=hidden_size, party_nums=len(base_model_dict), output_dim=num_classes)

        sl_model = SLModelTF(
            base_model_dict=base_model_dict, device_y=device_y,  model_fuse=model_fuse)
        sl_model.fit(data, label,
                     epochs=1, batch_size=128, shuffle=True)
        global_metric = sl_model.evaluate(data, label, batch_size=128)
        zero_metric = sl_model.evaluate(
            data, label, sample_weight=sample_weights, batch_size=128)
        # 测试验证 sample_weight有效性
        self.assertEquals(zero_metric['loss'], 0.0)
        self.assertGreater(global_metric['accuracy'], 0.9)
        result = sl_model.predict(data, batch_size=128, verbose=1)
        self.assertIsNotNone(result)
        base_model_path = os.path.join(_temp_dir, "base_model")
        fuse_model_path = os.path.join(_temp_dir, "fuse_model")
        sl_model.save_model(base_model_path=base_model_path,
                            fuse_model_path=fuse_model_path, is_test=True)
        self.assertIsNotNone(os.path.exists(base_model_path))
        self.assertIsNotNone(os.path.exists(fuse_model_path))

        sl_model.load_model(base_model_path=base_model_path,
                            fuse_model_path=fuse_model_path, is_test=True)
        reload_metric = sl_model.evaluate(
            data, label,  batch_size=128)
        self.assertEquals(global_metric, reload_metric)
