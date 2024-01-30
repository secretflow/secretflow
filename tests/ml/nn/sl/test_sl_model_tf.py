#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
import logging
import math
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from secretflow.data.ndarray import load
from secretflow.device import reveal
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.sl.agglayer.agg_method import Average
from secretflow.security.privacy import DPStrategy, LabelDP
from secretflow.security.privacy.mechanism.tensorflow import GaussianEmbeddingDP
from secretflow.utils.compressor import MixedCompressor, QuantizedZeroPoint, TopkSparse
from secretflow.utils.simulation.datasets import load_mnist

_temp_dir = tempfile.mkdtemp()


def create_dataset_builder(
    batch_size=32,
    repeat_count=5,
):
    def dataset_builder(x):
        import pandas as pd
        import tensorflow as tf

        x = [t.values if isinstance(t, pd.DataFrame) else t for t in x]
        x = x[0] if len(x) == 1 else tuple(x)
        data_set = (
            tf.data.Dataset.from_tensor_slices(x).batch(batch_size).repeat(repeat_count)
        )

        return data_set

    return dataset_builder


def create_base_model(input_dim, output_dim, output_num, name='base_model', l2=None):
    # Create model
    def create_model():
        from tensorflow import keras

        inputs = keras.Input(shape=input_dim)
        conv = keras.layers.Conv2D(filters=2, kernel_size=(3, 3))(inputs)
        pooling = keras.layers.MaxPooling2D()(conv)
        flatten = keras.layers.Flatten()(pooling)
        dropout = keras.layers.Dropout(0.5)(flatten)
        regularizer = keras.regularizers.L2(l2=l2) if l2 else None
        output_layers = [
            keras.layers.Dense(output_dim, kernel_regularizer=regularizer)(dropout)
            for _ in range(output_num)
        ]

        model = keras.Model(inputs, output_layers)

        # Compile model
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
        return model  # need wrap

    return create_model


def create_fuse_model(input_dim, output_dim, party_nums, input_num, name='fuse_model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers

        # input
        input_layers = []
        for i in range(party_nums * input_num):
            input_layers.append(
                keras.Input(
                    input_dim,
                )
            )
        # user define hidden process logic
        merged_layer = layers.concatenate(input_layers)
        fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
        output = layers.Dense(output_dim, activation='softmax')(fuse_layer)
        # Create model
        model = keras.Model(inputs=input_layers, outputs=output)
        # Compile model
        model.compile(
            loss=['categorical_crossentropy'],
            optimizer='adam',
            metrics=["accuracy"],
        )
        return model

    return create_model


def create_fuse_model_agglayer(input_dim, output_dim, name='fuse_model'):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers

        input_layer = keras.Input(input_dim)
        fuse_layer = layers.Dense(64, activation='relu')(input_layer)
        output = layers.Dense(output_dim, activation='softmax')(fuse_layer)
        # Create model
        model = keras.Model(inputs=input_layer, outputs=output)
        # Compile model
        model.compile(
            loss=['categorical_crossentropy'],
            optimizer='adam',
            metrics=["accuracy"],
        )
        return model

    return create_model


num_classes = 10
input_shape = (28, 28, 1)
hidden_size = 64
train_batch_size = 64
num_samples = 1000


def keras_model_with_mnist(
    devices,
    base_model_dict,
    device_y,
    model_fuse,
    data,
    label,
    strategy='split_nn',
    acc_threshold=0.7,
    eval_batch_size=128,
    **kwargs,
):
    # kwargs parsing
    dp_strategy_dict = kwargs.get('dp_strategy_dict', None)
    dataset_builder = kwargs.get('dataset_builder', None)

    base_local_steps = kwargs.get('base_local_steps', 1)
    fuse_local_steps = kwargs.get('fuse_local_steps', 1)
    bound_param = kwargs.get('bound_param', 0.0)

    loss_thres = kwargs.get('loss_thres', 0.01)
    split_steps = kwargs.get('split_steps', 1)
    max_fuse_local_steps = kwargs.get('max_fuse_local_steps', 10)

    agg_method = kwargs.get('agg_method', None)
    compressor = kwargs.get('compressor', None)
    pipeline_size = kwargs.get('pipeline_size', 1)

    party_shape = data.partition_shape()
    data_length = party_shape[devices.alice][0]

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=model_fuse,
        dp_strategy_dict=dp_strategy_dict,
        simulation=True,
        random_seed=1234,
        backend="tensorflow",
        strategy=strategy,
        base_local_steps=base_local_steps,
        fuse_local_steps=fuse_local_steps,
        bound_param=bound_param,
        loss_thres=loss_thres,
        split_steps=split_steps,
        max_fuse_local_steps=max_fuse_local_steps,
        agg_method=agg_method,
        compressor=compressor,
        pipeline_size=pipeline_size,
    )

    history = sl_model.fit(
        data,
        label,
        validation_data=(data, label),
        epochs=2,
        batch_size=train_batch_size,
        shuffle=False,
        random_seed=1234,
        dataset_builder=dataset_builder,
        audit_log_dir=_temp_dir,
        audit_log_params={'save_format': 'h5'},
    )
    global_metric = sl_model.evaluate(
        data,
        label,
        batch_size=eval_batch_size,
        random_seed=1234,
        dataset_builder=dataset_builder,
    )

    sample_weights = load({devices.bob: devices.bob(lambda: np.zeros(data_length))()})
    zero_metric = sl_model.evaluate(
        data,
        label,
        sample_weight=sample_weights,
        batch_size=eval_batch_size,
        random_seed=1234,
        dataset_builder=dataset_builder,
    )
    # test history
    assert math.isclose(
        global_metric['accuracy'], history['val_accuracy'][-1], rel_tol=0.02
    )
    loss_sum = 0
    for device, worker in sl_model._workers.items():
        if device not in base_model_dict:
            continue
        loss = reveal(worker.get_base_losses())
        if len(loss) > 0:
            import tensorflow as tf

            loss_sum += tf.add_n(loss).numpy()
    if loss_sum != 0:
        assert math.isclose(zero_metric['loss'], loss_sum, rel_tol=0.01)
    else:
        assert zero_metric['loss'] == 0.0
    assert global_metric['accuracy'] > acc_threshold
    result = sl_model.predict(data, batch_size=128, verbose=1)
    reveal_result = []
    for rt in result:
        reveal_result.extend(reveal(rt))
    assert len(reveal_result) == data_length
    base_model_path = os.path.join(_temp_dir, "base_model")
    fuse_model_path = os.path.join(_temp_dir, "fuse_model")
    sl_model.save_model(
        base_model_path=base_model_path,
        fuse_model_path=fuse_model_path,
        is_test=True,
    )
    assert os.path.exists(base_model_path)
    assert os.path.exists(fuse_model_path)

    reload_base_model_dict = {}
    for device in base_model_dict.keys():
        reload_base_model_dict[device] = None

    sl_model_load = SLModel(
        base_model_dict=reload_base_model_dict,
        device_y=device_y,
        model_fuse=model_fuse,
        dp_strategy_dict=dp_strategy_dict,
        simulation=True,
        random_seed=1234,
        strategy=strategy,
        base_local_steps=base_local_steps,
        fuse_local_steps=fuse_local_steps,
        bound_param=bound_param,
        agg_method=agg_method,
        compressor=compressor,
    )
    sl_model_load.load_model(
        base_model_path=base_model_path,
        fuse_model_path=fuse_model_path,
        is_test=True,
    )
    reload_metric = sl_model_load.evaluate(
        data,
        label,
        batch_size=eval_batch_size,
        random_seed=1234,
        dataset_builder=dataset_builder,
    )
    assert math.isclose(
        global_metric['accuracy'], reload_metric['accuracy'], rel_tol=0.01
    )
    assert math.isclose(global_metric['loss'], reload_metric['loss'], rel_tol=0.1)

    def _assert_tensor_info(tensor_info):
        assert tensor_info['inputs']
        assert tensor_info['outputs']
        assert len(tensor_info['inputs']) > 0
        assert len(tensor_info['outputs']) > 0

    export_tf_base_path = os.path.join(_temp_dir, "base_model_export_tf")
    export_tf_fuse_path = os.path.join(_temp_dir, "fuse_model_export_tf")
    tensor_infos = sl_model_load.export_model(
        base_model_path=export_tf_base_path,
        fuse_model_path=export_tf_fuse_path,
        is_test=True,
    )
    base_tensor_infos, fuse_tensor_info = reveal(tensor_infos)
    print(base_tensor_infos, fuse_tensor_info)
    for _, base_tensor_info in base_tensor_infos.items():
        _assert_tensor_info(base_tensor_info)
    _assert_tensor_info(fuse_tensor_info)

    export_onnx_base_path = os.path.join(_temp_dir, "base_model_export_onnx")
    export_onnx_fuse_path = os.path.join(_temp_dir, "fuse_model_export_onnx")
    tensor_infos = sl_model_load.export_model(
        base_model_path=export_onnx_base_path,
        fuse_model_path=export_onnx_fuse_path,
        is_test=True,
    )

    base_tensor_infos, fuse_tensor_info = reveal(tensor_infos)
    print(base_tensor_infos, fuse_tensor_info)
    for _, base_tensor_info in base_tensor_infos.items():
        _assert_tensor_info(base_tensor_info)
    _assert_tensor_info(fuse_tensor_info)


@pytest.fixture(scope='module')
def sf_mnist_data(sf_simulation_setup_devices):
    (x_train, y_train), (_, _) = load_mnist(
        parts={
            sf_simulation_setup_devices.alice: (0, num_samples),
            sf_simulation_setup_devices.bob: (0, num_samples),
        },
        normalized_x=True,
        categorical_y=True,
    )

    @dataclass
    class Prepared:
        x_train = None
        y_train = None

    prepared = Prepared()
    prepared.x_train = x_train
    prepared.y_train = y_train

    yield prepared
    del prepared


@pytest.fixture(scope='module')
def sf_single_output_model(sf_simulation_setup_devices):
    global _temp_dir
    _temp_dir = reveal(sf_simulation_setup_devices.alice(lambda: tempfile.mkdtemp())())
    # keras model
    base_model = create_base_model(input_shape, 64, output_num=1)
    base_model_dict = {
        sf_simulation_setup_devices.alice: base_model,
        sf_simulation_setup_devices.bob: base_model,
    }
    fuse_model = create_fuse_model(
        input_dim=hidden_size,
        input_num=1,
        party_nums=len(base_model_dict),
        output_dim=num_classes,
    )

    @dataclass
    class Prepared:
        base_model_dict = None
        fuse_model = None

    prepared = Prepared()
    prepared.base_model_dict = base_model_dict
    prepared.fuse_model = fuse_model

    yield prepared
    del prepared


@pytest.fixture(scope='module')
def sf_multi_output_model(sf_simulation_setup_devices):
    global _temp_dir
    _temp_dir = reveal(sf_simulation_setup_devices.alice(lambda: tempfile.mkdtemp())())
    # prepare model
    basenet_output = 2

    # keras model
    base_model = create_base_model(input_shape, 64, output_num=basenet_output)
    base_model_dict = {
        sf_simulation_setup_devices.alice: base_model,
        sf_simulation_setup_devices.bob: base_model,
    }
    fuse_model = create_fuse_model(
        input_dim=hidden_size,
        input_num=basenet_output,
        party_nums=len(base_model_dict),
        output_dim=num_classes,
    )

    @dataclass
    class Prepared:
        base_model_dict = None
        fuse_model = None

    prepared = Prepared()
    prepared.base_model_dict = base_model_dict
    prepared.fuse_model = fuse_model

    yield prepared
    del prepared


@pytest.fixture(scope='module')
def sf_single_feature_model(sf_simulation_setup_devices):
    global _temp_dir
    _temp_dir = reveal(sf_simulation_setup_devices.alice(lambda: tempfile.mkdtemp())())
    # keras model
    base_model = create_base_model(input_shape, 64, output_num=1)
    base_model_dict = {
        sf_simulation_setup_devices.alice: base_model,
    }
    fuse_model = create_fuse_model_agglayer(
        input_dim=hidden_size,
        output_dim=num_classes,
    )

    @dataclass
    class Prepared:
        base_model_dict = None
        fuse_model = None

    prepared = Prepared()
    prepared.base_model_dict = base_model_dict
    prepared.fuse_model = fuse_model

    yield prepared
    del prepared


class TestSLModelTensorflow:
    def test_dp_strategy(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_output_model
    ):
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
        dp_strategy_dict = {
            sf_simulation_setup_devices.alice: dp_strategy_alice,
            sf_simulation_setup_devices.bob: dp_strategy_bob,
        }
        dp_spent_step_freq = 10

        print("test dp strategy")
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            dp_strategy_dict=dp_strategy_dict,
            dp_spent_step_freq=dp_spent_step_freq,
            acc_threshold=0.68,
        )

    def test_dataset_builder(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_output_model
    ):
        # test dataset builder
        print("test Dataset builder")
        dataset_buidler_dict = {
            sf_simulation_setup_devices.alice: create_dataset_builder(
                batch_size=train_batch_size,
                repeat_count=2,
            ),
            sf_simulation_setup_devices.bob: create_dataset_builder(
                batch_size=train_batch_size,
                repeat_count=2,
            ),
        }
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            dataset_builder=dataset_buidler_dict,
            eval_batch_size=train_batch_size,
        )

    def test_split_async_strategy(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_output_model
    ):
        print("test split async strategy")
        # test split async with multiple base local steps
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='split_async',
            base_local_steps=5,
            fuse_local_steps=1,
        )
        # test split async with multiple fuse local steps
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='split_async',
            base_local_steps=1,
            fuse_local_steps=5,
            bound_param=0.1,
        )
        # test split async with both base and fuse multiple local steps
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='split_async',
            base_local_steps=5,
            fuse_local_steps=5,
            bound_param=0.1,
        )

    def test_split_state_async_strategy(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_output_model
    ):
        # test split state async
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='split_state_async',
            loss_thres=0.01,
            split_steps=1,
            max_fuse_local_steps=10,
        )

    def test_pipeline_strategy(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_output_model
    ):
        # test split state async
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='pipeline',
            pipeline_size=2,
        )

    def test_model_with_regularizer(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_output_model
    ):
        # test model with regularizer
        base_model_with_reg = create_base_model(input_shape, 64, output_num=1, l2=1e-3)
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict={
                sf_simulation_setup_devices.alice: base_model_with_reg,
                sf_simulation_setup_devices.bob: base_model_with_reg,
            },
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
        )

    def test_model_with_topk_compressor(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_output_model
    ):
        print("test TopK Sparse")
        top_k_compressor = TopkSparse(0.5)
        keras_model_with_mnist(
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            devices=sf_simulation_setup_devices,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            compressor=top_k_compressor,
        )

    def test_model_with_quantized_compressor(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_output_model
    ):
        print("test quantized compressor")
        quantized_compressor = QuantizedZeroPoint()
        keras_model_with_mnist(
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            devices=sf_simulation_setup_devices,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            compressor=quantized_compressor,
        )

    def test_model_with_mixed_compressor(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_output_model
    ):
        print("test mixed compressor")
        mixed_compressor = MixedCompressor(QuantizedZeroPoint(), TopkSparse(0.5))
        keras_model_with_mnist(
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            devices=sf_simulation_setup_devices,
            base_model_dict=sf_single_output_model.base_model_dict,
            model_fuse=sf_single_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            compressor=mixed_compressor,
        )

    def test_multi_output_model(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_multi_output_model
    ):
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_multi_output_model.base_model_dict,
            model_fuse=sf_multi_output_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
        )

    def test_secure_agglayer(self, sf_simulation_setup_devices, sf_mnist_data):
        base_model = create_base_model(input_shape, 64, output_num=1)
        base_model_dict = {
            sf_simulation_setup_devices.alice: base_model,
            sf_simulation_setup_devices.bob: base_model,
        }
        fuse_model = create_fuse_model_agglayer(
            input_dim=hidden_size,
            output_dim=num_classes,
        )

        print("test AggLayer")
        keras_model_with_mnist(
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            agg_method=Average(),
        )

    def test_secure_agglayer_with_compressor(
        self, sf_simulation_setup_devices, sf_mnist_data
    ):
        base_model = create_base_model(input_shape, 64, output_num=1)
        base_model_dict = {
            sf_simulation_setup_devices.alice: base_model,
            sf_simulation_setup_devices.bob: base_model,
        }
        fuse_model = create_fuse_model_agglayer(
            input_dim=hidden_size,
            output_dim=num_classes,
        )

        # agg layer
        print("test PlainAggLayer")
        keras_model_with_mnist(
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            devices=sf_simulation_setup_devices,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            agg_method=Average(),
            compressor=TopkSparse(0.5),
            acc_threshold=0.65,
        )

    def test_single_feature_with_agg_layer(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_feature_model
    ):
        # agg layer
        print("test PlainAggLayer with topk sparse")
        top_k_compressor = TopkSparse(0.5)
        keras_model_with_mnist(
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            devices=sf_simulation_setup_devices,
            base_model_dict=sf_single_feature_model.base_model_dict,
            model_fuse=sf_single_feature_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            agg_method=Average(),
            compressor=top_k_compressor,
            acc_threshold=0.65,
        )
        print("test PlainAggLayer with quantized compressor")
        quantized_compressor = QuantizedZeroPoint()
        keras_model_with_mnist(
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            devices=sf_simulation_setup_devices,
            base_model_dict=sf_single_feature_model.base_model_dict,
            model_fuse=sf_single_feature_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            agg_method=Average(),
            compressor=quantized_compressor,
            acc_threshold=0.65,
        )

    def test_single_feature_with_dp(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_feature_model
    ):
        print("test dp strategy")
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_feature_model.base_model_dict,
            model_fuse=sf_single_feature_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
        )

    @pytest.mark.skip(reason='fixme: base backward error with wrong gradient shape')
    def test_single_feature_with_topk_compressor(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_feature_model
    ):
        print("test single feature with topk sparse")
        top_k_compressor = TopkSparse(0.5)
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_feature_model.base_model_dict,
            model_fuse=sf_single_feature_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            compressor=top_k_compressor,
        )

    def test_single_feature_with_quantized_compressor(
        self, sf_simulation_setup_devices, sf_mnist_data, sf_single_feature_model
    ):
        print("test single featurewith quantized compressor")
        quantized_compressor = QuantizedZeroPoint()
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=sf_mnist_data.x_train,
            label=sf_mnist_data.y_train,
            base_model_dict=sf_single_feature_model.base_model_dict,
            model_fuse=sf_single_feature_model.fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            compressor=quantized_compressor,
        )


def random_csv_data(alice, bob, multi_labels=False) -> dict:
    train_feature = np.random.uniform(0, 1, [1024, 8])
    coef = np.random.uniform(0, 1, [8])
    _, alice_path = tempfile.mkstemp()
    _, bob_path = tempfile.mkstemp()
    pd.DataFrame(train_feature[:, :4]).to_csv(alice_path, index=False)
    bob_df = pd.DataFrame(train_feature[:, 4:]).rename(columns={0: 4, 1: 5, 2: 6, 3: 7})
    label_df = pd.DataFrame(coef)
    bob_df["label"] = label_df
    if multi_labels:
        bob_df['label2'] = label_df
    bob_df.to_csv(bob_path, index=False)
    return {
        alice: alice_path,
        bob: bob_path,
    }


class TestSLModelTensorflowFileInput:
    def create_base_model(self, col_names: list):
        # Create model
        def create_model():
            import tensorflow as tf
            from tensorflow import keras

            np.random.seed(7)
            input_layers = []
            for i in range(4):
                input_layers.append(keras.Input(shape=(1,), name=col_names[i]))
            merged_layer = keras.layers.concatenate(input_layers)
            out_layer = keras.layers.Dense(
                1,
                activation=None,
                use_bias=False,
            )(merged_layer)
            model = keras.Model(inputs=input_layers, outputs=out_layer)
            # Compile model
            model.summary()
            model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(),
                metrics=[tf.keras.metrics.AUC()],
            )
            return model

        return create_model

    def create_fuse_model(self):
        def create_model():
            import tensorflow as tf
            from tensorflow import keras

            #  input
            input_layers = [
                keras.Input(1),
                keras.Input(1),
            ]
            output = tf.nn.sigmoid(input_layers[0] + input_layers[1])
            model = keras.Model(inputs=input_layers, outputs=output)

            model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(),
                metrics=[tf.keras.metrics.AUC()],
            )
            return model

        return create_model

    def create_dataset_builder(self, num_labels=0):
        def dataset_builder(x):
            import tensorflow as tf

            label = None
            if num_labels > 0:
                labels = x[1] if isinstance(x[1], (list, tuple)) else [x[1]]
                logging.warning(
                    f"x1 = {x[1]}, labels = {labels} len(labels) == num_labels = {len(labels) == num_labels}"
                )
                assert len(labels) == num_labels and all(
                    isinstance(lb, str) for lb in labels
                )
                label = x[1][0] if isinstance(x[1], (list, tuple)) else x[1]
            data_set = tf.data.experimental.make_csv_dataset(
                x[0],
                batch_size=train_batch_size,
                label_name=label,
                header=True,
                num_epochs=1,
            )
            return data_set

        return dataset_builder

    def demo_model_with_demo_data(
        self, devices, label_names=None, dataset_builder_dict=None
    ):
        base_model_dict = {
            devices.alice: self.create_base_model(col_names=["0", "1", "2", "3"]),
            devices.bob: self.create_base_model(col_names=["4", "5", "6", "7"]),
        }
        model_fuse = self.create_fuse_model()
        sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=devices.bob,
            model_fuse=model_fuse,
            simulation=True,
            random_seed=1234,
            backend="tensorflow",
        )
        multi_labels = True if isinstance(label_names, (list, tuple)) else False
        file_pathes = random_csv_data(devices.alice, devices.bob, multi_labels)
        sl_model.fit(
            file_pathes,
            label_names,
            validation_data=(file_pathes, label_names),
            epochs=2,
            batch_size=64,
            shuffle=False,
            random_seed=1234,
            dataset_builder=dataset_builder_dict,
        )

    def test_file_input_single_label(self, sf_simulation_setup_devices):
        self.demo_model_with_demo_data(sf_simulation_setup_devices, "label")

    def test_file_input_single_label_with_dataset_builder(
        self, sf_simulation_setup_devices
    ):
        dataset_builder_dict = {
            sf_simulation_setup_devices.alice: self.create_dataset_builder(
                num_labels=0
            ),
            sf_simulation_setup_devices.bob: self.create_dataset_builder(num_labels=1),
        }
        self.demo_model_with_demo_data(
            sf_simulation_setup_devices,
            'label',
            dataset_builder_dict=dataset_builder_dict,
        )

    def test_file_input_multi_label_with_dataset_builder(
        self, sf_simulation_setup_devices
    ):
        dataset_builder_dict = {
            sf_simulation_setup_devices.alice: self.create_dataset_builder(
                num_labels=0
            ),
            sf_simulation_setup_devices.bob: self.create_dataset_builder(num_labels=2),
        }
        self.demo_model_with_demo_data(
            sf_simulation_setup_devices,
            ['label', 'label2'],
            dataset_builder_dict=dataset_builder_dict,
        )
