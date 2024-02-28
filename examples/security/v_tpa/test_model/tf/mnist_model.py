#!/usr/bin/env python
# coding=utf-8
import sys

sys.path.append("..")

import pdb

import numpy as np
import tensorflow as tf
from tensorflow import keras, nn, optimizers
from tensorflow.keras import layers


def create_passive_model(input_shape, output_shape, opt_args, compile_args):
    def create():
        class PassiveModel(keras.Model):
            def __init__(self, input_shape, output_shape):
                super().__init__()

                self.in_shape = input_shape
                self.out_shape = output_shape
                self.linear1 = layers.Dense(32, activation="relu")
                self.linear2 = layers.Dense(self.out_shape)

            def call(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        input_feature = keras.Input(shape=input_shape)
        output = PassiveModel(input_shape, output_shape)(input_feature)
        model = keras.Model(inputs=input_feature, outputs=output)

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


def get_passive_model(input_shape, output_shape, opt_args, compile_args):
    return create_passive_model(input_shape, output_shape, opt_args, compile_args)
