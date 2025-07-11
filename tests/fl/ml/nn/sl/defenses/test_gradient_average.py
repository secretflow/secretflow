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
import torch
from torch import nn
from torchmetrics import Accuracy

from secretflow.data.split import train_test_split
from secretflow.preprocessing import StandardScaler
from secretflow.utils.simulation.datasets import load_creditcard_small
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.defenses.gradient_average import GradientAverage


def test_gradient_average_tensorflow_backend(sf_simulation_setup_devices):

    data = load_creditcard_small(
        {sf_simulation_setup_devices.alice: (0, 29)}, num_sample=5000
    )
    label = load_creditcard_small(
        {sf_simulation_setup_devices.bob: (29, 30)}, num_sample=5000
    ).astype(np.float32)
    scaler = StandardScaler()
    data = scaler.fit_transform(data).astype(np.float32)
    random_state = 1234
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state
    )
    train_label, test_label = train_test_split(
        label, train_size=0.8, random_state=random_state
    )
    hidden_dim_1 = 16

    def create_base_net(input_dim, hidden_dim, name="first_net"):
        # Create model
        def create_model():
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            model = keras.Sequential(
                [
                    keras.Input(shape=input_dim),
                    layers.Dense(hidden_dim, activation="relu"),
                    layers.Dense(hidden_dim, activation="relu"),
                ],
                name=name,
            )
            # Compile model
            model.summary()
            optimizer = tf.keras.optimizers.Adam()
            model.compile(
                loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
            return model

        return create_model

    def create_fuse_model(input_dim_1, output_dim, party_nums, name="fuse_model"):
        def create_model():
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            # input
            input_layers = keras.Input(
                input_dim_1,
            )
            middle_layer = layers.Dense(input_dim_1 // 2)(input_layers)
            output = layers.Dense(output_dim, activation="sigmoid")(middle_layer)

            model = keras.Model(
                inputs=input_layers,
                outputs=output,
                name=name,
            )
            model.summary()
            optimizer = tf.keras.optimizers.Adam()

            model.compile(
                loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
            return model

        return create_model

    base_model_dict = {
        sf_simulation_setup_devices.alice: create_base_net(
            input_dim=29, hidden_dim=hidden_dim_1
        ),
    }
    fuse_model = create_fuse_model(input_dim_1=hidden_dim_1, party_nums=2, output_dim=1)

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=sf_simulation_setup_devices.bob,
        model_fuse=fuse_model,
        simulation=True,
        random_seed=1234,
        strategy="split_nn",
    )

    gradient_average = GradientAverage(backend="tensorflow")
    history = sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=1,
        batch_size=128,
        shuffle=False,
        random_seed=1234,
        callbacks=[gradient_average],
    )

    assert history["val_accuracy"][-1] > 0.6


def test_gradient_average_torch_backend(sf_simulation_setup_devices):

    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    data = load_creditcard_small({alice: (0, 29)}, num_sample=5000)
    label = load_creditcard_small({bob: (29, 30)}, num_sample=5000).astype(np.float32)
    scaler = StandardScaler()
    data = scaler.fit_transform(data).astype(np.float32)
    random_state = 1234

    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state
    )
    train_label, test_label = train_test_split(
        label, train_size=0.8, random_state=random_state
    )
    base_model = TorchModel(
        model_fn=DnnBase,
        loss_fn=nn.BCELoss,
        optim_fn=optim_wrapper(torch.optim.Adam),
        metrics=[
            metric_wrapper(Accuracy, task="binary"),
        ],
        input_dims=[29],
        dnn_units_size=[16],
    )
    fuse_model = TorchModel(
        model_fn=DnnFuse,
        loss_fn=nn.BCELoss,
        optim_fn=optim_wrapper(torch.optim.Adam),
        metrics=[
            metric_wrapper(Accuracy, task="binary"),
        ],
        input_dims=[16],
        dnn_units_size=[1],
    )
    base_model_dict = {
        alice: base_model,
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=bob,
        model_fuse=fuse_model,
        simulation=True,
        random_seed=1234,
        strategy="split_nn",
        backend="torch",
    )
    gradient_average = GradientAverage(backend="torch")

    history = sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=1,
        batch_size=128,
        shuffle=False,
        random_seed=1234,
        callbacks=[gradient_average],
    )
    assert history["val_BinaryAccuracy"][-1] > 0.6
