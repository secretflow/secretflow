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

import pdb

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras, nn, optimizers
from tensorflow.keras import layers


class Identity(keras.layers.Layer):
    def __init__(self, num_outputs, activation=None):
        super().__init__()
        self.num_outputs = num_outputs
        self.activation = activation

    def build(self, input_shape):
        pass

    def call(self, inputs):
        if self.activation is None:
            return inputs
        elif self.activation == "relu":
            return tf.nn.relu(inputs)
        elif self.activation == "softmax":
            return tf.nn.softmax(inputs)
        elif self.activation == "sigmoid":
            return tf.nn.sigmoid(inputs)
        else:
            return inputs


def create_fuse_model_naive(input_shapes, output_shape, opt_args, compile_args):
    def create():
        class FuseModel(keras.Model):
            def __init__(self, input_shapes, output_shapes):
                super().__init__()

                self.in_shapes = input_shapes
                # self.out_shapes = output_shapes
                # self.identity = Identity(output_shape, activation='softmax')

            def call(self, xs):
                x = layers.add(xs)
                # x = self.identity(x)
                return tf.nn.softmax(x, axis=1)

        input_layers = [keras.Input(input_shape) for input_shape in input_shapes]
        output = FuseModel(input_shapes, output_shape)(input_layers)
        model = keras.Model(inputs=input_layers, outputs=output)

        optimizer = tf.keras.optimizers.get(
            {
                "class_name": opt_args.get("class_name", "sgd"),
                "config": opt_args["config"],
            }
        )

        model.compile(
            loss=compile_args["loss"],
            optimizer=optimizer,
            metrics=compile_args["metrics"],
        )

        return model

    return create


def create_fuse_model_sum(input_shapes, output_shape, opt_args, compile_args):
    def create():
        class FuseModel(keras.Model):
            def __init__(self, input_shapes, output_shapes):
                super().__init__()

                self.in_shapes = input_shapes
                self.out_shapes = output_shapes
                self.linear = layers.Dense(output_shape, activation="softmax")

            def call(self, xs):
                x = layers.add(xs)
                x = self.linear(x)
                return x

        input_layers = [keras.Input(input_shape) for input_shape in input_shapes]
        output = FuseModel(input_shapes, output_shape)(input_layers)
        model = keras.Model(inputs=input_layers, outputs=output)

        optimizer = tf.keras.optimizers.get(
            {
                "class_name": opt_args.get("class_name", "sgd"),
                "config": opt_args["config"],
            }
        )

        model.compile(
            loss=compile_args["loss"],
            optimizer=optimizer,
            metrics=compile_args["metrics"],
        )

        return model

    return create


def create_fuse_model_average(input_shapes, output_shape, opt_args, compile_args):
    def create():
        class FuseModel(keras.Model):
            def __init__(self, input_shapes, output_shapes):
                super().__init__()

                self.in_shapes = input_shapes
                self.out_shapes = output_shapes
                self.linear = layers.Dense(output_shape, activation="softmax")

            def call(self, xs):
                x = layers.average(xs)
                x = self.linear(x)
                return x

        input_layers = [keras.Input(input_shape) for input_shape in input_shapes]
        output = FuseModel(input_shapes, output_shape)(input_layers)
        model = keras.Model(inputs=input_layers, outputs=output)

        optimizer = tf.keras.optimizers.get(
            {
                "class_name": opt_args.get("class_name", "sgd"),
                "config": opt_args["config"],
            }
        )

        model.compile(
            loss=compile_args["loss"],
            optimizer=optimizer,
            metrics=compile_args["metrics"],
        )

        return model

    return create


def create_fuse_model_cat(input_shapes, output_shape, opt_args, compile_args):
    def create():
        # input_layers = [keras.Input(input_shape) for input_shape in input_shapes]
        # merged_layer = layers.concatenate(input_layers)
        # output = layers.Dense(output_shape, activation='softmax')(merged_layer)
        # model = keras.Model(inputs=input_layers, outputs=output)
        class FuseModel(keras.Model):
            def __init__(self, input_shapes, output_shapes):
                super().__init__()

                self.in_shapes = input_shapes
                self.out_shapes = output_shapes
                self.linear = layers.Dense(output_shape, activation="softmax")

            def call(self, xs):
                x = layers.concatenate(xs)
                x = self.linear(x)
                return x

        input_layers = [keras.Input(input_shape) for input_shape in input_shapes]
        output = FuseModel(input_shapes, output_shape)(input_layers)
        model = keras.Model(inputs=input_layers, outputs=output)

        optimizer = tf.keras.optimizers.get(
            {
                "class_name": opt_args.get("class_name", "sgd"),
                "config": opt_args["config"],
            }
        )

        model.compile(
            loss=compile_args["loss"],
            optimizer=optimizer,
            metrics=compile_args["metrics"],
        )
        return model

    return create


def get_fuse_model(input_shapes, output_shape, aggregation, opt_args, compile_args):
    if aggregation == "naive_sum":
        model = create_fuse_model_naive(
            input_shapes, output_shape, opt_args, compile_args
        )
    elif aggregation == "sum":
        model = create_fuse_model_sum(
            input_shapes, output_shape, opt_args, compile_args
        )
    elif aggregation == "average":
        model = create_fuse_model_average(
            input_shapes, output_shape, opt_args, compile_args
        )
    elif aggregation == "concatenate":
        model = create_fuse_model_cat(
            input_shapes, output_shape, opt_args, compile_args
        )
    else:
        raise TypeError("Invalid aggregation method!!!")
    return model
