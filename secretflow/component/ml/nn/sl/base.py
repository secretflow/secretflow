# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from dataclasses import asdict, dataclass, fields
from typing import Dict, List

import secretflow as sf
from secretflow import PYU
from secretflow.component.data_utils import BaseEnum

# current version 0.1
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 1


@dataclass
class ModelMeta:
    parts: List
    model_input_scheme: str
    label_col: List[str]
    feature_names: List[str]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict):
        field_set = {f.name for f in fields(cls) if f.init}
        filtered_data = {k: v for k, v in data.items() if k in field_set}
        return cls(**filtered_data)


@enum.unique
class ModelInputScheme(BaseEnum):
    TENSOR = "tensor"
    TENSOR_DICT = "tensor_dict"

    @staticmethod
    def values():
        return [str(value) for value in ModelInputScheme.__members__.values()]


def mkdtemp(pyus: List[PYU]):
    import tempfile

    return {pyu: sf.reveal(pyu(tempfile.mkdtemp)()) for pyu in pyus}


BUILTIN_OPTIMIZERS = [
    "Adam",
    "SGD",
    "RMSprop",
    "AdamW",
    "Adamax",
    "Nadam",
    "Adagrad",
    "Adadelta",
    "Adafactor",
    "Ftrl",
    "Lion",
]

BUILTIN_LOSSES = [
    "binary_crossentropy",
    "categorical_crossentropy",
    "mean_squared_error",
    "mean_squared_logarithmic_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "cosine_similarity",
    "huber",
    "kl_divergence",
    "log_cosh",
    "poisson",
    "binary_focal_crossentropy",
    "sparse_categorical_crossentropy",
    "hinge",
    "categorical_hinge",
    "squared_hinge",
]

BUILTIN_METRICS = [
    "AUC",
    "Accuracy",
    "Precision",
    "Recall",
    "BinaryAccuracy",
    "BinaryCrossentropy",
    "CategoricalAccuracy",
    "CategoricalCrossentropy",
    "CosineSimilarity",
    "FalseNegatives",
    "FalsePositives",
    "TrueNegatives",
    "TruePositives",
    "KLDivergence",
    "LogCoshError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanRelativeError",
    "MeanSquaredError",
    "MeanSquaredLogarithmicError",
    "Hinge",
    "SquaredHinge",
    "CategoricalHinge",
    "BinaryIoU",
    "IoU",
    "MeanIoU",
    "OneHotIoU",
    "OneHotMeanIoU",
    "Poisson",
    "PrecisionAtRecall",
    "RecallAtPrecision",
    "RootMeanSquaredError",
    "SensitivityAtSpecificity",
    "SparseCategoricalAccuracy",
    "SparseCategoricalCrossentropy",
    "SparseTopKCategoricalAccuracy",
    "SpecificityAtSensitivity",
    "TopKCategoricalAccuracy",
]

BUILTIN_STRATEGIES = [
    "pipeline",
    "split_nn",
    "split_async",
    "split_state_async",
]

BUILTIN_COMPRESSORS = [
    "",
    "topk_sparse",
    "random_sparse",
    "stc_sparse",
    "scr_sparse",
    "quantized_fp",
    "quantized_lstm",
    "quantized_kmeans",
    "quantized_zeropoint",
    "mixed_compressor",
]

DEFAULT_MODELS_CODE = '''\
# pre imported:
# import tensorflow as tf
# from tensorflow import Module, keras
# from tensorflow.keras import Model, layers
# from tensorflow.keras.layers import Layer
# from secretflow.ml.nn import applications as apps

def create_base_model(input_dim, output_dim):
    model = keras.Sequential(
        [
            keras.Input(shape=input_dim),
            layers.Dense(100, activation="relu"),
            layers.Dense(output_dim, activation="relu"),
        ]
    )
    return model

def create_fuse_model(input_dim):
    input_layers = [keras.Input(input_dim), keras.Input(input_dim)]
    merged_layer = layers.concatenate(input_layers)
    fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
    output = layers.Dense(1, activation='sigmoid')(fuse_layer)
    return keras.Model(inputs=input_layers, outputs=output)

hidden_size = 64

fit(
    client_base=create_base_model(12, hidden_size),
    server_base=create_base_model(4, hidden_size),
    server_fuse=create_fuse_model(hidden_size),
)
'''

DEFAULT_CUSTOM_LOSS_CODE = '''\
def loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


compile_loss(loss)

'''
