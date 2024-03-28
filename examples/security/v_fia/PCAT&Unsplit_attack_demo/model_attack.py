import collections

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models


def create_base_model(input_dim, output_dim, name="base_model"):
    # Create model
    def create_model():
        model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(100, activation="relu"),
                layers.Dense(output_dim, activation="relu"),
            ]
        )
        # Compile model
        model.summary()
        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model


def create_fuse_model(input_dim, output_dim, party_nums, name="fuse_model"):
    def create_model():
        # input
        input_layers = []
        for i in range(party_nums):
            input_layers.append(
                keras.Input(
                    input_dim,
                )
            )

        merged_layer = layers.concatenate(input_layers)
        fuse_layer = layers.Dense(64, activation="relu")(merged_layer)
        output = layers.Dense(output_dim, activation="sigmoid")(fuse_layer)

        model = keras.Model(inputs=input_layers, outputs=output)
        model.summary()

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model


def create_pcat_base_model(
    image_size, in_dim, output_dim, name="create_pcat_base_model"
):
    def create_testnet():
        model = keras.Sequential(
            [
                layers.Input(shape=(image_size, image_size, in_dim)),
                layers.Conv2D(64, 3, 2, padding="same"),
                layers.Conv2D(128, 3, 2, padding="same"),
                layers.Conv2D(output_dim, 3, 1, padding="same"),
            ]
        )

        model.summary()
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_testnet


def create_pcat_attack_decoder_model(
    input_shape, name="create_pcat_attack_decoder_model"
):
    def create_testnet():
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv2DTranspose(256, 3, 2, padding="same", activation="relu"),
                layers.Conv2DTranspose(128, 3, 2, padding="same", activation="relu"),
                layers.Conv2D(3, 3, 1, padding="same", activation="tanh"),
            ]
        )

        model.summary()
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_testnet


def create_server_model(input_shape, output_dim, name="create_server_model"):
    def create_model():
        # input
        input_layers = []
        input_layers.append(
            keras.Input(
                input_shape,
            )
        )
        # for i in range(party_nums):
        #    input_layers.append(keras.Input(input_dim, ))
        merged_layer = layers.concatenate(input_layers)
        f_layer = layers.Flatten()(merged_layer)
        output1 = layers.Dense(64, activation="softmax")(f_layer)
        output2 = layers.Dense(output_dim, activation="softmax")(output1)

        model = keras.Model(inputs=input_layers, outputs=output2)

        model.summary()

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model


def create_unsplit_base_model(
    in_dim, output_dim, flag, name="create_unsplit_base_model"
):
    def create_testnet():
        model = keras.Sequential(
            [
                layers.Input(shape=(28, 28, in_dim)),
                layers.Conv2D(
                    filters=output_dim,
                    kernel_size=5,
                    strides=1,
                    padding="same",
                ),
                layers.ReLU(),
                layers.MaxPool2D(pool_size=2, strides=2),
            ]
        )

        model.summary()
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    def create_testnet_2():
        model = keras.Sequential(
            [
                layers.Input(shape=(28, 28, in_dim)),
                layers.Conv2D(
                    filters=output_dim,
                    kernel_size=5,
                    strides=1,
                    padding="same",
                ),
                layers.ReLU(),
            ]
        )

        model.summary()
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    if flag == 0:
        return create_testnet
    else:
        return create_testnet_2


def create_unsplit_server_model(
    input_shape, output_dim, name="create_unsplit_server_model"
):
    def create_model():
        # input
        input_layers = []
        input_layers.append(
            keras.Input(
                input_shape,
            )
        )
        # for i in range(party_nums):
        #    input_layers.append(keras.Input(input_dim, ))
        merged_layer = layers.concatenate(input_layers)
        f_layer = layers.Flatten()(merged_layer)
        output1 = layers.Dense(64, activation="softmax")(f_layer)
        output2 = layers.Dense(output_dim, activation="softmax")(output1)

        model = keras.Model(inputs=input_layers, outputs=output2)

        model.summary()

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model
