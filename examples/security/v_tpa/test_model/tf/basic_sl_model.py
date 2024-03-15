import copy
import pdb

import numpy as np
import tensorflow as tf
from tensorflow import keras, nn, optimizers
from tensorflow.keras import layers


def create_passive_model(
    input_shape, n_class, SplitModel, split_point, party_num, opt_args, compile_args
):
    def create():
        class Model(keras.Model):
            def __init__(self, input_shape):
                super().__init__()

                self.in_shape = input_shape
                self.model, _ = SplitModel(n_class).split(split_point, party_num)

            def call(self, x):
                x = self.model(x)
                return x

        input_feature = keras.Input(shape=input_shape)
        m = Model(input_shape)
        output = m(input_feature)
        m = keras.Model(inputs=input_feature, outputs=output)

        # keras optimizer, legacy
        """
        optimizer = tf.keras.optimizers.get({
            'class_name': opt_args.get('class_name', 'adam'),
            'config': opt_args['config']
        })
        """
        opt_name = opt_args.get("class_name", "sgd")
        lr = opt_args["config"].get("learning_rate", 0.01)
        if opt_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif opt_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise "Invalid Optimizer {}!!!".format(opt_name)

        m.compile(
            loss=compile_args["loss"],
            optimizer=optimizer,
            metrics=compile_args["metrics"],
        )
        return m

    return create


def create_fuse_model(
    input_shapes,
    agg,
    n_class,
    SplittableModel,
    split_point,
    party_num,
    opt_args,
    compile_args,
):
    def create():
        class FuseModel(keras.Model):
            def __init__(self, input_shapes):
                super().__init__()

                self.in_shapes = input_shapes
                _, self.model = SplittableModel(n_class).split(split_point, party_num)

            def call(self, xs):
                if agg == "sum":
                    x = layers.add(xs)
                elif agg == "average":
                    x = layers.average(xs)
                elif agg == "concatenate":
                    x = layers.concatenate(xs)
                else:
                    raise "Invalid aggregatio {}!!!".format(agg)
                x = self.model(x)
                return tf.nn.softmax(x, axis=1)

        input_layers = [keras.Input(input_shape) for input_shape in input_shapes]
        m = FuseModel(input_shapes)
        output = m(input_layers)
        m = keras.Model(inputs=input_layers, outputs=output)

        opt_name = opt_args.get("class_name", "sgd")
        lr = opt_args["config"].get("learning_rate", 0.01)
        if opt_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif opt_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise "Invalid Optimizer {}!!!".format(opt_name)

        m.compile(
            loss=compile_args["loss"],
            optimizer=optimizer,
            metrics=compile_args["metrics"],
        )

        return m

    return create
