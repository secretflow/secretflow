"""bank marketing models

The following codes are demos only. 
It's **NOT for production** due to system security concerns,
please **DO NOT** use it directly in production.
"""

import os
from random import seed
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = Tuple[XYList, XYList]

os.environ['PYTHONHASHSEED'] = str(1)
seed(1)
tf.random.set_seed(1)
np.random.seed(1)


def create_bank_model(input_dim, output_dim, opt):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_dim),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='sigmoid'),
        ]
    )
    model.summary()
    model.compile(
        loss='binary_crossentropy', optimizer=opt, metrics=["accuracy", 'AUC']
    )
    return model


def preprocess(data, processor, columns: List):
    for i in columns:
        data[i] = processor.fit_transform(data[i])
    return data


def create_partitioned_dataset(
    data: np.ndarray, clients_num: int, split_dimension: List[int] = None
):
    if split_dimension is None:
        part_dim = data.shape[1] // clients_num
        split_dimension = [part_dim * (i + 1) for i in range(clients_num - 1)]
        split_data = np.split(data, split_dimension, axis=1)
    else:
        assert len(split_dimension) == (clients_num - 1)
        split_data = np.split(data, split_dimension, axis=1)

    return split_data


def get_split_dimension(dim, n):
    split_dimension = [0]
    part_dim = dim // n
    for i in range(n - 1):
        split_dimension.append(part_dim * (i + 1))
    split_dimension.append(dim)
    return split_dimension


def create_base_model(input_dim, output_dim, name='base_model'):
    # Create model
    def create_model():
        model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(128, activation="relu"),
                layers.Dense(output_dim, activation="relu"),
            ]
        )
        # Compile model
        model.summary()
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model  # 不能序列化的

    return create_model


# 创建fuse模型
def create_fuse_model(input_dim, output_dim, party_nums, name='fuse_model'):
    def create_model():
        # input
        input_layers = []
        for i in range(party_nums):
            input_layers.append(
                keras.Input(
                    input_dim,
                )
            )

        # 定义融合逻辑
        merged_layer = layers.concatenate(input_layers)
        fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
        output = layers.Dense(output_dim, activation='sigmoid')(fuse_layer)
        # 构建模型
        model = keras.Model(inputs=input_layers, outputs=output)
        model.summary()
        # 编译模型，定义损失，优化器，以及指标
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model
