# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import tensorflow as tf


class DnnBase(tf.keras.Model):
    def __init__(
        self,
        dnn_units_size: List[int],
        dnn_units_activation: List[str],
        preprocess_layer: Optional[tf.keras.Model] = None,
        embedding_dim=16,
        **kwargs,
    ):
        assert len(dnn_units_size) == len(dnn_units_activation), (
            f"dnn_units_size len ({len(dnn_units_size)}) "
            f"must equal with dnn_units_activation len ({len(dnn_units_activation)})"
        )
        super(DnnBase, self).__init__(**kwargs)
        self._preprocess_layer = preprocess_layer
        if self._preprocess_layer:
            outputs = self._preprocess_layer.output
            self._embedding_features_layer = [
                tf.keras.layers.Dense(units=embedding_dim) for _ in range(len(outputs))
            ]
        self._sparse_features_layer = tf.keras.layers.Concatenate()
        self._dense_layers = [
            tf.keras.layers.Dense(dnn_units_size[i], activation=dnn_units_activation[i])
            for i in range(len(dnn_units_size))
        ]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        if self._preprocess_layer:
            x = self._preprocess_layer(x, training=True)
            embedding_features = []
            for i, d in enumerate(x):
                embedding = self._embedding_features_layer[i](d)
                embedding_features.append(embedding)
            x = tf.concat(embedding_features, axis=1)
        for dense_layer in self._dense_layers:
            x = dense_layer(x)
        return x

    def output_num(self):
        return 1


class DnnFuse(tf.keras.Model):
    def __init__(
        self,
        input_shapes: List[int],
        dnn_units_size: List[int],
        dnn_units_activation: List[str],
        *args,
        **kwargs,
    ):
        assert len(dnn_units_size) == len(dnn_units_activation), (
            f"dnn_units_size len ({len(dnn_units_size)}) "
            f"must equal with dnn_units_activation len ({len(dnn_units_activation)})"
        )
        super(DnnFuse, self).__init__(*args, **kwargs)
        self._input_length = len(input_shapes)
        self._dense_layers = [
            tf.keras.layers.Dense(dnn_units_size[i], activation=dnn_units_activation[i])
            for i in range(len(dnn_units_size))
        ]

    def call(self, inputs, training=None, mask=None):
        assert (
            len(inputs) == self._input_length
        ), f"the input nums not match, need {self._input_length}, got {len(inputs)}"
        x = tf.concat(inputs, axis=-1)
        for dense_layer in self._dense_layers:
            x = dense_layer(x)
        return x
