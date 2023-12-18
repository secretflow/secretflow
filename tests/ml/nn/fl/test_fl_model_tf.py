#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""

import functools
import os
import tempfile

import numpy as np
import tensorflow as tf

from secretflow.data.ndarray import load
from secretflow.device import reveal
from secretflow.ml.nn import FLModel
from secretflow.ml.nn.fl.compress import COMPRESS_STRATEGY
from secretflow.preprocessing.encoder import OneHotEncoder
from secretflow.security.aggregation import PlainAggregator, SparsePlainAggregator
from secretflow.security.compare import PlainComparator
from secretflow.security.privacy import DPStrategyFL, GaussianModelDP
from secretflow.utils.simulation.datasets import load_iris, load_mnist

_temp_dir = tempfile.mkdtemp()

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

path_to_flower_dataset = tf.keras.utils.get_file(
    "flower_photos",
    "https://secretflow-data.oss-accelerate.aliyuncs.com/datasets/tf_flowers/flower_photos.tgz",
    untar=True,
    cache_dir=_temp_dir,
)


# model define for mlp
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


# model define for cnn
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


# model define for flower recognaiton
def create_conv_flower_model(input_shape, num_classes, name='model'):
    def create_model():
        from tensorflow import keras

        # Create model

        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        # Compile model
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=["accuracy"],
        )
        return model

    return create_model


class TestFedModelDF:
    def test_keras_model_for_iris(self, sf_simulation_setup_devices):
        """unittest ignore"""
        aggregator = PlainAggregator(sf_simulation_setup_devices.carol)
        comparator = PlainComparator(sf_simulation_setup_devices.carol)
        hdf = load_iris(
            parts=[sf_simulation_setup_devices.alice, sf_simulation_setup_devices.bob],
            aggregator=aggregator,
            comparator=comparator,
        )

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

        device_list = [
            sf_simulation_setup_devices.alice,
            sf_simulation_setup_devices.bob,
        ]
        fed_model = FLModel(
            device_list=device_list,
            model=model,
            aggregator=aggregator,
            sampler="batch",
            random_seed=1234,
        )
        fed_model.fit(data, label, epochs=1, batch_size=16, aggregate_freq=3)
        global_metric, _ = fed_model.evaluate(data, label, batch_size=16)
        print(global_metric)

        # FIXME(fengjun.feng): This assert is failing.
        # assert global_metric[1].result().numpy() > 0.7


class TestFedModelCSV:
    def test_keras_model(self, sf_simulation_setup_devices):
        aggregator = PlainAggregator(sf_simulation_setup_devices.carol)
        train_data = load_iris(
            parts=[sf_simulation_setup_devices.alice, sf_simulation_setup_devices.bob],
            aggregator=aggregator,
        )
        _, alice_path = tempfile.mkstemp()
        _, bob_path = tempfile.mkstemp()
        train_path = {
            sf_simulation_setup_devices.alice: alice_path,
            sf_simulation_setup_devices.bob: bob_path,
        }
        train_data.to_csv(train_path, index=False)

        valid_path = train_path

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
        model = create_nn_model(n_features, n_classes, 8, 3)
        device_list = [
            sf_simulation_setup_devices.alice,
            sf_simulation_setup_devices.bob,
        ]
        fed_model = FLModel(device_list=device_list, model=model, aggregator=aggregator)

        fed_model.fit(
            train_path,
            "class",
            epochs=1,
            validation_data=valid_path,
            validation_freq=1,
            label_decoder=onehot_func,
            batch_size=32,
            aggregate_freq=2,
        )
        global_metric, _ = fed_model.evaluate(
            valid_path,
            "class",
            label_decoder=onehot_func,
            batch_size=16,
        )
        print(global_metric[1].result().numpy())


class TestFedModelTensorflow:
    def keras_model_with_mnist(
        self,
        devices,
        model,
        data,
        label,
        strategy,
        backend,
        aggregator=None,
        **kwargs,
    ):
        if not aggregator:
            if strategy in COMPRESS_STRATEGY:
                aggregator = SparsePlainAggregator(devices.carol)
            else:
                aggregator = PlainAggregator(devices.carol)
        else:
            aggregator = aggregator
        party_shape = data.partition_shape()
        alice_length = party_shape[devices.alice][0]
        bob_length = party_shape[devices.bob][0]

        alice_arr = devices.alice(lambda: np.zeros(alice_length))()
        bob_arr = devices.bob(lambda: np.zeros(bob_length))()
        sample_weights = load({devices.alice: alice_arr, devices.bob: bob_arr})

        # spcify params
        sampler_method = kwargs.get('sampler_method', "batch")
        dp_spent_step_freq = kwargs.get('dp_spent_step_freq', None)
        server_agg_method = kwargs.get("server_agg_method", None)
        device_list = [devices.alice, devices.bob]
        num_gpus = kwargs.get("num_gpus", 0)
        dp_strategy = kwargs.get("dp_strategy", None)

        fed_model = FLModel(
            server=devices.carol,
            device_list=device_list,
            model=model,
            aggregator=aggregator,
            backend=backend,
            strategy=strategy,
            random_seed=1234,
            server_agg_method=server_agg_method,
            num_gpus=num_gpus,
            dp_strategy=dp_strategy,
        )
        random_seed = 1524
        history = fed_model.fit(
            data,
            label,
            validation_data=(data, label),
            epochs=1,
            batch_size=128,
            aggregate_freq=2,
            sampler_method=sampler_method,
            random_seed=random_seed,
            dp_spent_step_freq=dp_spent_step_freq,
        )
        global_metric, _ = fed_model.evaluate(
            data,
            label,
            batch_size=128,
            sampler_method=sampler_method,
            random_seed=random_seed,
        )
        assert (
            global_metric[1].result().numpy()
            == history["global_history"]['val_accuracy'][-1]
        )

        assert (
            global_metric[1].result().numpy() > 0.1
        )  # just test functionality correctness.
        zero_metric, _ = fed_model.evaluate(
            data,
            label,
            sample_weight=sample_weights,
            batch_size=128,
            sampler_method=sampler_method,
            random_seed=random_seed,
        )
        # test sample_weight validation
        assert zero_metric[0].result() == 0.0
        result = fed_model.predict(data, batch_size=128, random_seed=random_seed)
        assert len(reveal(result[devices.alice])) == alice_length

        model_path_test = os.path.join(_temp_dir, "base_model")
        fed_model.save_model(model_path=model_path_test, is_test=True)
        model_path_dict = {
            devices.alice: os.path.join(_temp_dir, "alice_model"),
            devices.bob: os.path.join(_temp_dir, "bob_model"),
        }
        fed_model.save_model(model_path=model_path_dict, is_test=False)

        # test load model
        fed_model.load_model(model_path=model_path_test, is_test=True)
        fed_model.load_model(model_path=model_path_dict, is_test=False)
        reload_metric, _ = fed_model.evaluate(
            data,
            label,
            batch_size=128,
            sampler_method=sampler_method,
            random_seed=random_seed,
        )
        np.testing.assert_equal(
            [m.result().numpy() for m in global_metric],
            [m.result().numpy() for m in reload_metric],
        )

    def test_keras_model(self, sf_simulation_setup_devices):
        (_, _), (mnist_data, mnist_label) = load_mnist(
            parts={
                sf_simulation_setup_devices.alice: 0.4,
                sf_simulation_setup_devices.bob: 0.6,
            },
            normalized_x=True,
            categorical_y=True,
        )
        # prepare model
        num_classes = 10

        input_shape = (28, 28, 1)
        # keras model
        model = create_conv_model(input_shape, num_classes)

        # test fed avg w with possion sampler
        self.keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=mnist_data,
            label=mnist_label,
            model=model,
            strategy="fed_avg_w",
            backend="tensorflow",
            sampler_method='possion',
        )

        # test fed avg u test default batch sampler
        self.keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=mnist_data,
            label=mnist_label,
            model=model,
            strategy="fed_avg_u",
            backend="tensorflow",
        )
        # test fed prox
        self.keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=mnist_data,
            label=mnist_label,
            model=model,
            strategy="fed_prox",
            backend="tensorflow",
            mu=0.1,
        )
        # test fed stc
        self.keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=mnist_data,
            label=mnist_label,
            model=model,
            strategy="fed_stc",
            backend="tensorflow",
            sparsity=0.9,
        )
        # test fed scr
        self.keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=mnist_data,
            label=mnist_label,
            model=model,
            strategy="fed_scr",
            backend="tensorflow",
        )

        # test fed avg g with DP
        gaussian_model_gdp = GaussianModelDP(
            noise_multiplier=0.001,
            l2_norm_clip=0.1,
            num_clients=2,
            is_secure_generator=False,
        )
        dp_strategy_fl = DPStrategyFL(model_gdp=gaussian_model_gdp)
        dp_spent_step_freq = 10
        self.keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=mnist_data,
            label=mnist_label,
            model=model,
            strategy="fed_avg_g",
            dp_strategy=dp_strategy_fl,
            dp_spent_step_freq=dp_spent_step_freq,
            backend="tensorflow",
        )

        def create_server_agg_method():
            def server_agg_method(model_params_list):
                return model_params_list

            return server_agg_method

        server_agg_method = create_server_agg_method()
        self.keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=mnist_data,
            label=mnist_label,
            model=model,
            strategy="fed_avg_g",
            backend="tensorflow",
            aggregator=None,
            server_agg_method=server_agg_method,
        )


class TestFedModelDataLoader:
    def keras_model_with_mnist(
        self, devices, model, data, label, strategy, backend, **kwargs
    ):
        if strategy in COMPRESS_STRATEGY:
            aggregator = SparsePlainAggregator(devices.carol)
        else:
            aggregator = PlainAggregator(devices.carol)

        # spcify params
        sampler_method = kwargs.get('sampler_method', "batch")
        dp_spent_step_freq = kwargs.get('dp_spent_step_freq', None)
        dataset_builder = kwargs.get('dataset_builder', None)
        device_list = [devices.alice, devices.bob]

        fed_model = FLModel(
            server=devices.carol,
            device_list=device_list,
            model=model,
            aggregator=aggregator,
            backend=backend,
            strategy=strategy,
            random_seed=1234,
            **kwargs,
        )
        random_seed = 1234
        history = fed_model.fit(
            data,
            label,
            validation_data=data,
            epochs=1,
            batch_size=32,
            aggregate_freq=2,
            sampler_method=sampler_method,
            random_seed=random_seed,
            dp_spent_step_freq=dp_spent_step_freq,
            dataset_builder=dataset_builder,
        )
        global_metric, _ = fed_model.evaluate(
            data,
            label,
            batch_size=32,
            sampler_method=sampler_method,
            random_seed=random_seed,
            dataset_builder=dataset_builder,
        )
        assert (
            global_metric[1].result().numpy()
            == history["global_history"]['val_accuracy'][-1]
        )

        model_path_test = os.path.join(_temp_dir, "base_model")
        fed_model.save_model(model_path=model_path_test, is_test=True)
        model_path_dict = {
            devices.alice: os.path.join(_temp_dir, "alice_model"),
            devices.bob: os.path.join(_temp_dir, "bob_model"),
        }
        fed_model.save_model(model_path=model_path_dict, is_test=False)

        # test load model
        fed_model.load_model(model_path=model_path_test, is_test=True)
        fed_model.load_model(model_path=model_path_dict, is_test=False)
        reload_metric, _ = fed_model.evaluate(
            data,
            label,
            batch_size=32,
            sampler_method=sampler_method,
            random_seed=random_seed,
            dataset_builder=dataset_builder,
        )
        np.testing.assert_equal(
            [m.result().numpy() for m in global_metric],
            [m.result().numpy() for m in reload_metric],
        )

    def test_keras_model(self, sf_simulation_setup_devices):
        # prepare model
        num_classes = 5

        input_shape = (180, 180, 3)
        # keras model
        model = create_conv_flower_model(input_shape, num_classes)

        def create_dataset_builder(
            batch_size=32,
        ):
            def dataset_builder(x, stage="train"):
                import math

                import tensorflow as tf

                img_height = 180
                img_width = 180
                data_set = tf.keras.utils.image_dataset_from_directory(
                    x,
                    validation_split=0.2,
                    subset="both",
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size,
                )
                if stage == "train":
                    train_dataset = data_set[0]
                    train_step_per_epoch = math.ceil(
                        len(data_set[0].file_paths) / batch_size
                    )
                    return train_dataset, train_step_per_epoch
                elif stage == "eval":
                    eval_dataset = data_set[1]
                    eval_step_per_epoch = math.ceil(
                        len(data_set[1].file_paths) / batch_size
                    )
                    return eval_dataset, eval_step_per_epoch

            return dataset_builder

        data_builder_dict = {
            sf_simulation_setup_devices.alice: create_dataset_builder(
                batch_size=32,
            ),
            sf_simulation_setup_devices.bob: create_dataset_builder(
                batch_size=32,
            ),
        }

        # test fed avg w
        self.keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data={
                sf_simulation_setup_devices.alice: path_to_flower_dataset,
                sf_simulation_setup_devices.bob: path_to_flower_dataset,
            },
            label=None,
            model=model,
            strategy="fed_avg_w",
            backend="tensorflow",
            dataset_builder=data_builder_dict,
        )


class TestFedModelMemoryDF:
    def test_keras_model_for_iris(self, sf_memory_setup_devices):
        """unittest ignore"""
        aggregator = PlainAggregator(sf_memory_setup_devices.carol)
        comparator = PlainComparator(sf_memory_setup_devices.carol)
        hdf = load_iris(
            parts=[sf_memory_setup_devices.alice, sf_memory_setup_devices.bob],
            aggregator=aggregator,
            comparator=comparator,
        )

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

        device_list = [
            sf_memory_setup_devices.alice,
            sf_memory_setup_devices.bob,
        ]
        fed_model = FLModel(
            device_list=device_list,
            model=model,
            aggregator=aggregator,
            sampler="batch",
            random_seed=1234,
        )
        fed_model.fit(data, label, epochs=1, batch_size=16, aggregate_freq=3)
        global_metric, _ = fed_model.evaluate(data, label, batch_size=16)
        print(global_metric)

        # FIXME(fengjun.feng): This assert is failing.
        # assert global_metric[1].result().numpy() > 0.7
