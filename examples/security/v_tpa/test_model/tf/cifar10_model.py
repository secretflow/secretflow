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
sys.path.append("../..")

import pdb

import numpy as np
import tensorflow as tf
from model.tf.resnet_cifar import (
    ResNetCIFAR10,
    Splittable_ResNetCIFAR10,
    dist_weights,
    split_options,
)
from tensorflow import keras, nn, optimizers
from tensorflow.keras import layers


class Splittable_Model(Splittable_ResNetCIFAR10):
    def __init__(self, n_class):
        super().__init__(n_class)


def create_passive_model(input_shape, output_shape, opt_args, compile_args):
    def create():
        class PassiveModel(keras.Model):
            def __init__(self, input_shape, output_shape):
                super().__init__()

                self.in_shape = input_shape
                self.out_shape = output_shape
                self.model = ResNetCIFAR10(output_shape)

            def call(self, x):
                x = self.model(x)
                return x

        input_feature = keras.Input(shape=input_shape)
        pmodel = PassiveModel(input_shape, output_shape)
        output = pmodel(input_feature)
        model = keras.Model(inputs=input_feature, outputs=output)

        optimizer = tf.keras.optimizers.get(
            {
                "class_name": opt_args.get("class_name", "adam"),
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


def get_passive_model(input_shape, output_shape, opt_args, compile_args):
    return create_passive_model(input_shape, output_shape, opt_args, compile_args)


if __name__ == "__main__":
    input_shape = [16, 32, 3]
    output_shape = 10
    compile_args = {
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
    }
    opt_args = {
        "class_name": "adam",
        "config": {
            "learning_rate": 0.001,
        },
    }
    gen_model = create_passive_model(input_shape, output_shape, opt_args, compile_args)
    model = gen_model()
    pdb.set_trace()
    print("end")
