import sys

sys.path.append("..")

import copy
import pdb

import tensorflow as tf
import tensorflow.keras.backend as F
from tensorflow import keras, nn
from tensorflow.keras import layers


class Identity(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x = inputs * 1.0
        return x


class ResidualBlock(keras.Model):
    def __init__(self, out_channels, kernel_size, padding, strides):
        super().__init__()

        self.conv_res1 = layers.Conv2D(
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            use_bias=False,
        )
        self.conv_res1_bn = layers.BatchNormalization(momentum=0.9)
        self.conv_res2 = layers.Conv2D(
            out_channels, kernel_size=kernel_size, padding=padding, use_bias=False
        )
        self.conv_res2_bn = layers.BatchNormalization(momentum=0.9)

        if strides != 1:
            self.downsample = nn.Sequential(
                layers.Conv2D(
                    out_channels, kernel_size=1, strides=strides, use_bias=False
                ),
                layers.BatchNormalization(momentum=0.9),
            )
        else:
            self.downsample = None

        self.relu = layers.ReLU()

    def call(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out = out + residual
        return out


class ResNetCIFAR10(keras.Model):
    """
    A Residual network.
    """

    def __init__(self, n_class=10):
        super().__init__()

        self.conv = keras.Sequential(
            [
                layers.Conv2D(
                    32, kernel_size=3, strides=1, padding="same", use_bias=False
                ),
                layers.BatchNormalization(momentum=0.9),
                layers.ReLU(),
                layers.Conv2D(
                    64, kernel_size=3, strides=1, padding="same", use_bias=False
                ),
                layers.BatchNormalization(momentum=0.9),
                layers.ReLU(),
                ResidualBlock(
                    out_channels=64, kernel_size=3, strides=1, padding="same"
                ),
                layers.Conv2D(
                    128, kernel_size=3, strides=1, padding="same", use_bias=False
                ),
                layers.BatchNormalization(momentum=0.9),
                layers.ReLU(),
                layers.MaxPool2D((2, 2)),
                layers.Conv2D(
                    128, kernel_size=3, strides=1, padding="same", use_bias=False
                ),
                layers.BatchNormalization(momentum=0.9),
                layers.ReLU(),
                layers.MaxPool2D((2, 2)),
                ResidualBlock(
                    out_channels=128, kernel_size=3, strides=1, padding="same"
                ),
            ]
        )

        self.fc = keras.Sequential(
            [
                layers.Dense(128),
                layers.Dense(n_class),
            ]
        )

        self.avg_pool = layers.GlobalMaxPooling2D(keepdims=False)

    def call(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x


split_options = [3, 6, 11, 17]
dist_weights = [0.35, 0.35, 0.50, 1.5]


class Splittable_ResNetCIFAR10:
    """
    A Residual network.
    """

    def __init__(self, n_class=10):
        self.layers = [
            layers.Conv2D(32, kernel_size=3, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization(momentum=0.9),
            layers.ReLU(),
            layers.Conv2D(64, kernel_size=3, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization(momentum=0.9),
            layers.ReLU(),
            ResidualBlock(out_channels=64, kernel_size=3, strides=1, padding="same"),
            layers.Conv2D(
                128, kernel_size=3, strides=1, padding="same", use_bias=False
            ),
            layers.BatchNormalization(momentum=0.9),
            layers.ReLU(),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(
                128, kernel_size=3, strides=1, padding="same", use_bias=False
            ),
            layers.BatchNormalization(momentum=0.9),
            layers.ReLU(),
            layers.MaxPool2D((2, 2)),
            ResidualBlock(out_channels=128, kernel_size=3, strides=1, padding="same"),
            layers.GlobalMaxPooling2D(keepdims=False),
            layers.Dense(128),
            layers.Dense(n_class),
        ]

    def split(self, split_index, party_num):
        """
        if split_index in [3, 6]:
            bottom_model = keras.Sequential(self.layers[:split_index] + [Identity()])
        else:
            bottom_model = keras.Sequential(self.layers[:split_index])
        """

        bottom_model = keras.Sequential(copy.deepcopy(self.layers[:split_index]))
        fuse_model = keras.Sequential(copy.deepcopy(self.layers[split_index:]))
        return bottom_model, fuse_model


class Splittable_SimpleCIFAR10:
    pass


if __name__ == "__main__":
    # model = ResNetCIFAR10()
    split_cifar10 = Splittable_ResNetCIFAR10()
    bottom, fuse_model = split_cifar10.split(5, 2)
    pdb.set_trace()
    print("end")
