"""
The following codes are demos only. 
It's **NOT for production** due to system security concerns,
please **DO NOT** use it directly in production.
"""

import os
from random import seed
from typing import List, Tuple

import numpy as np
import tensorflow as tf

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = Tuple[XYList, XYList]

os.environ['PYTHONHASHSEED'] = str(1)
seed(1)
tf.random.set_seed(1)
np.random.seed(1)


def create_conv_model(input_shape, num_classes, opt, name='model'):
    # Create model
    def create_model():
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.summary()
        model.compile(
            loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"]
        )
        return model

    return create_model
