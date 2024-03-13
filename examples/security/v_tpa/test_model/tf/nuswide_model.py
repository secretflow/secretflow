#!/usr/bin/env python
# coding=utf-8
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

import sys

sys.path.append("..")

import numpy as np
import tensorflow as tf
from tensorflow import keras, nn, optimizers
from tensorflow.keras import layers


def create_passive_model(input_shape, output_shape, opt_args, compile_args):
    def create():
        class PassiveModel(keras.Model):
            def __init__(self, input_shape, output_shape):
                super().__init__()
                self.linear1 = layers.Dense(32, activation="relu")
                self.linear2 = layers.Dense(32, activation="relu")
                self.linear3 = layers.Dense(output_shape)

            def call(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        input_feature = keras.Input(shape=input_shape)
        output = PassiveModel(input_shape, output_shape)(input_feature)
        model = keras.Model(inputs=input_feature, outputs=output)

        optimizer = tf.keras.optimizers.get(
            {
                "class_name": opt_args.get("class_name", "sgd"),
                "config": {"learning_rate": opt_args.get("lr", 0.01)},
            }
        )

        model.compile(
            loss=compile_args["loss"],
            optimizer=optimizer,
            metrics=compile_args["metrics"],
        )
        return model

    return create


def get_passive_model(input_shape, output_shape, opt_args, compile_args):
    return create_passive_model(input_shape, output_shape, opt_args, compile_args)
