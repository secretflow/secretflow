#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""module docstring - short summary

If the description is long, the first line should be a short summary that makes sense on its own,
separated from the rest by a newline

"""
import math
import os
import tempfile

import numpy as np

from secretflow.data.ndarray import load
from secretflow.device import reveal
from secretflow.ml.nn import SLModel
from secretflow.security.privacy import DPStrategy, GaussianEmbeddingDP, LabelDP
from secretflow.utils.compressor import RandomSparse, TopkSparse
from secretflow.utils.simulation.datasets import load_mnist

_temp_dir = tempfile.mkdtemp()

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


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


num_classes = 10
input_shape = (28, 28, 1)
hidden_size = 64
train_batch_size = 128


def keras_model_with_mnist(
    devices,
    base_model_dict,
    device_y,
    model_fuse,
    data,
    label,
    strategy='split_nn',
    **kwargs
):
    # kwargs parsing
    dp_strategy_dict = kwargs.get('dp_strategy_dict', None)
    compressor = kwargs.get('compressor', None)
    dataset_builder = kwargs.get('dataset_builder', None)

    base_local_steps = kwargs.get('base_local_steps', 1)
    fuse_local_steps = kwargs.get('fuse_local_steps', 1)
    bound_param = kwargs.get('bound_param', 0.0)

    loss_thres = kwargs.get('loss_thres', 0.01)
    split_steps = kwargs.get('split_steps', 1)
    max_fuse_local_steps = kwargs.get('max_fuse_local_steps', 10)

    party_shape = data.partition_shape()
    alice_length = party_shape[devices.alice][0]
    bob_length = party_shape[devices.bob][0]

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=model_fuse,
        dp_strategy_dict=dp_strategy_dict,
        compressor=compressor,
        simulation=True,
        random_seed=1234,
        strategy=strategy,
        base_local_steps=base_local_steps,
        fuse_local_steps=fuse_local_steps,
        bound_param=bound_param,
        loss_thres=loss_thres,
        split_steps=split_steps,
        max_fuse_local_steps=max_fuse_local_steps,
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
        batch_size=128,
        random_seed=1234,
        dataset_builder=dataset_builder,
    )
    alice_arr = devices.alice(lambda: np.zeros(alice_length))()
    bob_arr = devices.bob(lambda: np.zeros(bob_length))()
    sample_weights = load({devices.alice: alice_arr, devices.bob: bob_arr})
    zero_metric = sl_model.evaluate(
        data,
        label,
        sample_weight=sample_weights,
        batch_size=128,
        random_seed=1234,
        dataset_builder=dataset_builder,
    )
    # test history
    assert math.isclose(
        global_metric['accuracy'], history['val_accuracy'][-1], rel_tol=0.01
    )
    loss_sum = 0
    for w in sl_model._workers.values():
        loss = reveal(w.get_base_losses())
        if len(loss) > 0:
            import tensorflow as tf

            loss_sum += tf.add_n(loss).numpy()
    if loss_sum != 0:
        assert math.isclose(zero_metric['loss'], loss_sum, rel_tol=0.01)
    else:
        assert zero_metric['loss'] == 0.0
    assert global_metric['accuracy'] > 0.8
    result = sl_model.predict(data, batch_size=128, verbose=1)
    reveal_result = []
    for rt in result:
        reveal_result.extend(reveal(rt))
    assert len(reveal_result) == alice_length
    base_model_path = os.path.join(_temp_dir, "base_model")
    fuse_model_path = os.path.join(_temp_dir, "fuse_model")
    sl_model.save_model(
        base_model_path=base_model_path,
        fuse_model_path=fuse_model_path,
        is_test=True,
    )
    assert os.path.exists(base_model_path)
    assert os.path.exists(fuse_model_path)

    sl_model_load = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=model_fuse,
        dp_strategy_dict=dp_strategy_dict,
        compressor=compressor,
        simulation=True,
        random_seed=1234,
        strategy=strategy,
        base_local_steps=base_local_steps,
        fuse_local_steps=fuse_local_steps,
        bound_param=bound_param,
    )
    sl_model_load.load_model(
        base_model_path=base_model_path,
        fuse_model_path=fuse_model_path,
        is_test=True,
    )
    reload_metric = sl_model_load.evaluate(
        data,
        label,
        batch_size=128,
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


class TestSLModelTensorflow:
    def test_single_output_model(self, sf_simulation_setup_devices):
        num_samples = 10000
        (x_train, y_train), (_, _) = load_mnist(
            parts={
                sf_simulation_setup_devices.alice: (0, num_samples),
                sf_simulation_setup_devices.bob: (0, num_samples),
            },
            normalized_x=True,
            categorical_y=True,
        )
        # prepare model
        num_classes = 10

        input_shape = (28, 28, 1)
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
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            dp_strategy_dict=dp_strategy_dict,
            dp_spent_step_freq=dp_spent_step_freq,
        )

        # test compressor
        print("test TopkSparse")
        top_k_compressor = TopkSparse(0.5)
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            compressor=top_k_compressor,
        )
        print("test RandomSparse")
        random_sparse = RandomSparse(0.5)
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            compressor=random_sparse,
        )

        # test dataset builder
        print("test Dataset builder")
        dataset_buidler_dict = {
            sf_simulation_setup_devices.alice: create_dataset_builder(
                batch_size=128,
                repeat_count=2,
            ),
            sf_simulation_setup_devices.bob: create_dataset_builder(
                batch_size=128,
                repeat_count=2,
            ),
        }
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            dataset_builder=dataset_buidler_dict,
        )
        print("test split async strategy")
        # test split async with multiple base local steps
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='split_async',
            base_local_steps=5,
            fuse_local_steps=1,
        )
        # test split async with multiple fuse local steps
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='split_async',
            base_local_steps=1,
            fuse_local_steps=5,
            bound_param=0.1,
        )
        # test split async with both base and fuse multiple local steps
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='split_async',
            base_local_steps=5,
            fuse_local_steps=5,
            bound_param=0.1,
        )
        # test split state async
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='split_state_async',
            loss_thres=0.01,
            split_steps=1,
            max_fuse_local_steps=10,
        )
        # test pipeline
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
            strategy='pipeline',
            pipeline_size=2,
        )
        # test model with regularizer
        base_model_with_reg = create_base_model(input_shape, 64, output_num=1, l2=1e-3)
        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict={
                sf_simulation_setup_devices.alice: base_model_with_reg,
                sf_simulation_setup_devices.bob: base_model_with_reg,
            },
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
        )

    def test_multi_output_model(self, sf_simulation_setup_devices):
        (x_train, y_train), (_, _) = load_mnist(
            parts={
                sf_simulation_setup_devices.alice: (0, 10000),
                sf_simulation_setup_devices.bob: (0, 10000),
            },
            normalized_x=True,
            categorical_y=True,
        )
        # prepare model
        num_classes = 10
        input_shape = (28, 28, 1)
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

        keras_model_with_mnist(
            devices=sf_simulation_setup_devices,
            data=x_train,
            label=y_train,
            base_model_dict=base_model_dict,
            model_fuse=fuse_model,
            device_y=sf_simulation_setup_devices.bob,
        )
