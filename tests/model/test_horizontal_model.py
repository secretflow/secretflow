#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""

import tempfile
import unittest
import os

import numpy as np

from secretflow.data.horizontal import read_csv
from secretflow.data.ndarray import (load)
from secretflow.model.fl_model import FLTFModel
from secretflow.preprocessing.encoder import OneHotEncoder
from secretflow.security.aggregation import PlainAggregator
from secretflow.security.compare import PlainComparator
from tests.basecase import DeviceTestCase

_temp_dir = tempfile.mkdtemp()

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


# 创建用户自定义模型
def create_nn_model(input_dim, output_dim, nodes, n=1, name='model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        # Create model
        model = keras.Sequential(name=name)
        for i in range(n):
            model.add(layers.Dense(
                nodes, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(output_dim, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy"])
        return model
    return create_model


# 创建用户自定义模型
def create_conv_model(input_shape, num_classes, name='model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers
        # Create model
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy"])
        return model

    return create_model


@unittest.skip("ignore this test, since iris is not a good dataset")
class TestFedModelDF(DeviceTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        path_alice = 'tests/datasets/iris/horizontal/iris.alice.csv'
        path_bob = 'tests/datasets/iris/horizontal/iris.bob.csv'

        cls.filepath = {cls.alice: path_alice, cls.bob: path_bob}

    def test_keras_model(self):
        aggregator = PlainAggregator(self.carol)
        comparator = PlainComparator(self.carol)
        hdf = read_csv(self.filepath, aggregator=aggregator,
                       comparator=comparator)

        label = hdf['class']
        # do preprocess
        encoder = OneHotEncoder()
        label = encoder.fit_transform(label)

        data = hdf.drop(columns='class', inplace=False)
        data = data.fillna(data.mean(numeric_only=True).to_dict())

        # prepare model
        n_features = 4
        n_classes = 3
        model = create_nn_model(n_features, n_classes, 8, 3)

        device_list = [self.alice, self.bob]
        fed_model = FLTFModel(
            device_list=device_list, model=model, aggregator=aggregator)
        fed_model.fit(data, label, epochs=100, batch_size=16,
                      aggregate_freq=3)
        global_metric = fed_model.evaluate(data, label, batch_size=16)
        self.assertGreater(global_metric[1], 0.7)


class TestFedModelNdArray(DeviceTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.path_x = 'tests/datasets/mnist/mnist_x_train.npy'
        cls.path_y = 'tests/datasets/mnist/mnist_y_train.npy'

    def test_keras_model(self):
        aggregator = PlainAggregator(self.carol)
        data = load({self.alice: self.path_x,
                     self.bob: self.path_x}, allow_pickle=True)
        label = load({self.alice: self.path_y,
                      self.bob: self.path_y}, allow_pickle=True)

        alice_arr = self.alice(lambda: np.zeros(60000))()
        bob_arr = self.bob(lambda: np.zeros(60000))()
        sample_weights = load({self.alice: alice_arr, self.bob: bob_arr})

        # prepare model
        num_classes = 10

        input_shape = (28, 28, 1)
        # 用户定义的已编译后的keras model
        model = create_conv_model(input_shape, num_classes)
        device_list = [self.alice, self.bob]
        fed_model = FLTFModel(
            device_list=device_list, model=model, aggregator=aggregator)
        fed_model.fit(data, label, epochs=1, batch_size=128, aggregate_freq=2)
        global_metric = fed_model.evaluate(data, label, batch_size=128)
        self.assertGreater(global_metric[1], 0.9)
        zero_metric = fed_model.evaluate(
            data, label, sample_weight=sample_weights, batch_size=128)
        # 测试验证 sample_weight有效性
        self.assertEquals(zero_metric[0], 0.0)
        model_path = os.path.join(_temp_dir, "base_model")
        fed_model.save_model(model_path=model_path, is_test=True)
        self.assertIsNotNone(os.path.exists(model_path))
        fed_model.load_model(model_path=model_path, is_test=True)
        reload_metric = fed_model.evaluate(data, label, batch_size=128)
        np.testing.assert_equal(global_metric, reload_metric)
