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

import tensorflow as tf
import torch
from torch import nn as nn
from torch import optim
from torchmetrics import AUROC, Accuracy, Precision

from secretflow_fl.ml.nn.core.torch import BaseModule


class ConvNetBase(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetBase, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )

        # fully connected layer, output 10 classes
        self.out = nn.Linear(192, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def output_num(self):
        return 1


class ConvNetFuse(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetFuse, self).__init__()
        self.fc1 = nn.Linear(64 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ConvNetFuseAgglayer(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetFuseAgglayer, self).__init__()
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ConvNetRegBase(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetRegBase, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )

        # fully connected layer, output 10 classes
        self.out = nn.Linear(192, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.sum(torch.abs(param))

        return output, reg_loss

    def output_num(self):
        return 2

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)

    def configure_metrics(self):
        """no use for sl base model."""
        return None

    def configure_loss(self):
        """no use for sl base model."""
        return None


class ConvNetRegFuse(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetRegFuse, self).__init__()
        self.fc1 = nn.Linear(64 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, hiddens):
        x = hiddens[::2]
        x = torch.cat(x, dim=1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx: int, dataloader_idx: int = 0, **kwargs):
        hiddens, y = batch
        out = self(hiddens)

        self.update_metrics(out, y)

        reg_loss = hiddens[1::2]
        reg_loss = reg_loss[0] + reg_loss[1]
        for param in self.parameters():
            reg_loss += torch.sum(torch.abs(param))

        loss = self.loss(out, y) + 1e-4 * reg_loss
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)

    def configure_metrics(self):
        return [
            Accuracy(task="multiclass", num_classes=10, average="micro"),
            Precision(task="multiclass", num_classes=10, average="micro"),
            AUROC(task="multiclass", num_classes=10),
        ]

    def configure_loss(self):
        return nn.CrossEntropyLoss()


def create_base_model(input_dim, output_dim, output_num, name="base_model", l2=None):
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
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model  # need wrap

    return create_model


def create_fuse_model(input_dim, output_dim, party_nums, input_num, name="fuse_model"):
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
        fuse_layer = layers.Dense(64, activation="relu")(merged_layer)
        output = layers.Dense(output_dim, activation="softmax")(fuse_layer)
        # Create model
        model = keras.Model(inputs=input_layers, outputs=output)
        # Compile model
        model.compile(
            loss=["categorical_crossentropy"],
            optimizer="adam",
            metrics=["accuracy"],
        )
        return model

    return create_model


def create_fuse_model_agglayer(input_dim, output_dim, name="fuse_model"):
    def create_model():
        from tensorflow import keras
        from tensorflow.keras import layers

        input_layer = keras.Input(input_dim)
        fuse_layer = layers.Dense(64, activation="relu")(input_layer)
        output = layers.Dense(output_dim, activation="softmax")(fuse_layer)
        # Create model
        model = keras.Model(inputs=input_layer, outputs=output)
        # Compile model
        model.compile(
            loss=["categorical_crossentropy"],
            optimizer="adam",
            metrics=["accuracy"],
        )
        return model

    return create_model


class FuseCustomLossModel(tf.keras.Model):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.fuse_layer = tf.keras.layers.Dense(64, activation="relu")
        self.output_layer = tf.keras.layers.Dense(output_dim, activation="softmax")

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    @property
    def metrics(self):
        return super().metrics + [self.loss_tracker]

    def reset_metrics(self):
        super().reset_metrics()
        self.loss_tracker.reset_state()

    def call(self, inputs, training=None):
        concat_inputs = tf.concat(inputs, axis=1)
        fused = self.fuse_layer(concat_inputs)
        out = self.output_layer(fused)
        # 1. you can calculate your loss in `call` and set compiled loss to None if needed.
        vars = self.trainable_variables
        reg_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in vars])
        self.add_loss(reg_loss)
        return out

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # 2. or override `compute_loss` to do custom loss calculation.
        loss = self.loss_fn(y, y_pred)
        self.loss_tracker.update_state(loss)
        if len(self.losses) > 0:
            loss += tf.add_n(self.losses)
        return loss

    def compute_metrics(self, x, y, y_pred, sample_weight):
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return self.get_metrics_result()

    def get_config(self):
        return {"output_dim": self.output_dim}


def create_fuse_model_custom_loss(
    input_dim, output_dim, party_nums, input_num, name="fuse_model"
):
    def create_model():
        import tensorflow as tf

        model = FuseCustomLossModel(output_dim)
        # Compile model
        model.compile(
            optimizer="adam",
            metrics=["accuracy"],
            loss=None,
        )
        # call to build
        model([tf.zeros(shape=(1, input_dim)) for _ in range(party_nums * input_num)])
        return model

    return create_model
