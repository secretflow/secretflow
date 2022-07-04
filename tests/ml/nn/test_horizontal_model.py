#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""

import functools
import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from secretflow.data.horizontal import read_csv
from secretflow.data.ndarray import load
from secretflow.ml.nn import FLModelTF
from secretflow.preprocessing.encoder import OneHotEncoder
from secretflow.security.aggregation import PlainAggregator
from secretflow.security.compare import PlainComparator
from tests.basecase import DeviceTestCase
from tests.utils.fed_dataset import load_iris_data, load_mnist_data

_temp_dir = tempfile.mkdtemp()

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


# 创建用户自定义模型
def create_nn_csv_model(input_dim, output_dim, nodes, n=1, name='model'):
    def create_model():
        inputs = {}
        for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
            inputs[feature] = tf.keras.Input(shape=(1,), name=feature)

        # Create model
        x = tf.keras.layers.Concatenate()(inputs.values())
        for i in range(n):
            x = tf.keras.layers.Dense(nodes, input_dim=input_dim, activation='relu')(x)
        x = tf.keras.layers.Dense(output_dim, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        # Compile model
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
        return model

    return create_model


# 创建用户自定义模型
def create_nn_model(input_dim, output_dim, nodes, n=1, name='model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers

        # Create model
        model = keras.Sequential(name=name)
        for i in range(n):
            model.add(layers.Dense(nodes, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(output_dim, activation='softmax'))

        # Compile model
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
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
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
        return model

    return create_model


@unittest.skip("ignore this test, since iris is not a good dataset")
class TestFedModelDF(DeviceTestCase):
    def test_keras_model(self):
        aggregator = PlainAggregator(self.carol)
        comparator = PlainComparator(self.carol)
        data_split = {
            self.alice: 0.5,
            self.bob: 1.0,
        }
        train_data = load_iris_data(party_ratio=data_split)

        hdf = read_csv(train_data, aggregator=aggregator, comparator=comparator)

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
        fed_model = FLModelTF(
            device_list=device_list, model=model, aggregator=aggregator, sampler="batch"
        )
        fed_model.fit(data, label, epochs=5, batch_size=16, aggregate_freq=3)
        global_metric, _ = fed_model.evaluate(data, label, batch_size=16)
        print(global_metric)
        self.assertGreater(global_metric[1].result().numpy(), 0.7)


class TestFedModelNdArray(DeviceTestCase):
    def test_keras_model(self):
        aggregator = PlainAggregator(self.carol)
        data_split = {
            self.alice: 0.4,
            self.bob: 1.0,
        }
        train_data_dict, _ = load_mnist_data(party_ratio=data_split)

        fed_npz = load(train_data_dict, allow_pickle=True)

        data = fed_npz['image']
        label = fed_npz['label']
        alice_arr = self.alice(lambda: np.zeros(24000))()
        bob_arr = self.bob(lambda: np.zeros(36000))()
        sample_weights = load({self.alice: alice_arr, self.bob: bob_arr})

        # prepare model
        num_classes = 10

        input_shape = (28, 28, 1)
        # 用户定义的已编译后的keras model
        model = create_conv_model(input_shape, num_classes)
        device_list = [self.alice, self.bob]
        fed_model = FLModelTF(
            device_list=device_list, model=model, aggregator=aggregator
        )
        history = fed_model.fit(
            data,
            label,
            validation_data=(data, label),
            epochs=5,
            batch_size=128,
            aggregate_freq=2,
        )
        global_metric, _ = fed_model.evaluate(data, label, batch_size=128)
        self.assertEquals(
            global_metric[1].result().numpy(),
            history.global_history['val_accuracy'][-1],
        )
        self.assertGreater(global_metric[1].result().numpy(), 0.9)
        zero_metric, _ = fed_model.evaluate(
            data, label, sample_weight=sample_weights, batch_size=128
        )
        # 测试验证 sample_weight有效性
        self.assertEquals(zero_metric[0].result(), 0.0)
        model_path = os.path.join(_temp_dir, "base_model")
        fed_model.save_model(model_path=model_path, is_test=True)
        self.assertIsNotNone(os.path.exists(model_path))

        # test load model
        new_fed_model = FLModelTF(
            device_list=device_list,
            model=None,
            aggregator=None,
        )
        new_fed_model.load_model(model_path=model_path, is_test=True)
        reload_metric, _ = new_fed_model.evaluate(data, label, batch_size=128)
        np.testing.assert_equal(
            [m.result().numpy() for m in global_metric],
            [m.result().numpy() for m in reload_metric],
        )


class TestFedModelNdArrayPossion(DeviceTestCase):
    def test_keras_model(self):
        aggregator = PlainAggregator(self.carol)
        data_split = {
            self.alice: 0.4,
            self.bob: 1.0,
        }
        train_data_dict, _ = load_mnist_data(party_ratio=data_split)

        fed_npz = load(train_data_dict, allow_pickle=True)

        data = fed_npz['image']
        label = fed_npz['label']
        alice_arr = self.alice(lambda: np.zeros(24000))()
        bob_arr = self.bob(lambda: np.zeros(36000))()
        sample_weights = load({self.alice: alice_arr, self.bob: bob_arr})

        # prepare model
        num_classes = 10

        input_shape = (28, 28, 1)
        # 用户定义的已编译后的keras model
        model = create_conv_model(input_shape, num_classes)
        device_list = [self.alice, self.bob]
        fed_model = FLModelTF(
            device_list=device_list,
            model=model,
            aggregator=aggregator,
            sampler='possion',
        )
        history = fed_model.fit(
            data,
            label,
            validation_data=(data, label),
            epochs=5,
            batch_size=128,
            aggregate_freq=2,
        )
        global_metric, _ = fed_model.evaluate(data, label, batch_size=128)
        self.assertEquals(
            global_metric[1].result().numpy(),
            history.global_history['val_accuracy'][-1],
        )
        self.assertGreater(global_metric[1].result().numpy(), 0.9)
        zero_metric, _ = fed_model.evaluate(
            data, label, sample_weight=sample_weights, batch_size=128
        )
        # test sample_weight
        self.assertEquals(zero_metric[0].result(), 0.0)
        model_path = os.path.join(_temp_dir, "base_model")
        fed_model.save_model(model_path=model_path, is_test=True)
        self.assertIsNotNone(os.path.exists(model_path))
        # test load model
        fed_model.load_model(model_path=model_path, is_test=True)
        reload_metric, _ = fed_model.evaluate(data, label, batch_size=128)
        np.testing.assert_equal(
            [m.result().numpy() for m in global_metric],
            [m.result().numpy() for m in reload_metric],
        )


class TestFedModelCSV(DeviceTestCase):
    def test_keras_model(self):
        aggregator = PlainAggregator(self.carol)
        train_data = {
            self.alice: "tests/datasets/iris/horizontal/iris.alice.csv",
            self.bob: "tests/datasets/iris/horizontal/iris.bob.csv",
        }
        valid_data = {
            self.alice: "tests/datasets/iris/horizontal/iris.alice.csv",
            self.bob: "tests/datasets/iris/horizontal/iris.bob.csv",
        }

        def label_decoder(x, label, num_class):
            # How to deal with label column
            vocab = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
            layer = tf.keras.layers.StringLookup(vocabulary=vocab)
            label = layer(label)
            one_hot_label = tf.one_hot(label, depth=num_class, axis=-1)
            return x, one_hot_label

        # prepare model
        n_features = 4
        n_classes = 3
        onehot_func = functools.partial(label_decoder, num_class=n_classes)
        model = create_nn_csv_model(n_features, n_classes, 8, 3)
        device_list = [self.alice, self.bob]
        fed_model = FLModelTF(
            device_list=device_list, model=model, aggregator=aggregator
        )

        fed_model.fit(
            train_data,
            "class",
            epochs=1,
            validation_data=valid_data,
            validation_freq=1,
            label_decoder=onehot_func,
            batch_size=32,
            aggregate_freq=2,
        )
        global_metric, _ = fed_model.evaluate(
            valid_data,
            "class",
            label_decoder=onehot_func,
            batch_size=16,
            verbose=0,
        )
        print(global_metric[1].result().numpy())
