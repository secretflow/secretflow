# Copyright 2022 Ant Group Co., Ltd.
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

import tensorflow as tf


class DeepFMbase(tf.keras.Model):
    def __init__(
        self,
        dnn_units_size,
        dnn_activation="relu",
        preprocess_layer=None,
        fm_embedding_dim=16,
        **kwargs,
    ):
        """Split learning version of DeepFM
        Args:
            dnn_units_size: list,list of positive integer or empty list, the layer number and units in each layer of DNN
            dnn_activation: activation function of dnn part
            preprocess_layer: The preprocessed layer a keras model, output a dict of preprocessed data
            fm_embedding_dim: fm embedding dim, default to be 16
        """

        super(DeepFMbase, self).__init__(**kwargs)

        self.preprocess = preprocess_layer
        outputs = self.preprocess.output

        self._sparse_features_layer = tf.keras.layers.Concatenate()
        self._embedding_features_layer = {
            k: tf.keras.layers.Dense(units=fm_embedding_dim) for k, v in outputs.items()
        }

        self._dnn_units_size = dnn_units_size
        self._dnn_activation = dnn_activation

        self._has_dense = False

        # FM Layer
        self._fm_first_order_sparse = tf.keras.layers.Dense(
            units=1, kernel_initializer="zeros", name="linear"
        )

        self.dense_internal = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=self._dnn_activation)
                for units in self._dnn_units_size
            ]
        )

        self._fm_first_order_dense = tf.keras.layers.Dense(
            units=1, kernel_initializer="zeros", name="linear"
        )

    def call(self, inputs, **kwargs):
        # preprocess net
        preprocess_data = self.preprocess(inputs, training=True)

        # base net
        embedding_features = []
        for column_name in preprocess_data:
            dense_features = self._embedding_features_layer.get(column_name)
            if dense_features is not None:
                embedding = dense_features(preprocess_data[column_name])
                embedding_features.append(embedding)
        sparse_features = self._sparse_features_layer(preprocess_data.values())

        stack_embeddings = tf.stack(embedding_features, axis=1)
        # stop gradient for FM way
        stack_embeddings = tf.stop_gradient(stack_embeddings)
        concat_embeddings = tf.concat(embedding_features, axis=1)
        first_order_fm = self._fm_first_order_sparse(sparse_features)

        base_out = self.dense_internal(concat_embeddings)
        x_sum = tf.reduce_sum(stack_embeddings, axis=1)
        x_square_sum = tf.reduce_sum(
            tf.reduce_sum(tf.pow(stack_embeddings, 2), axis=1), axis=1, keepdims=True
        )
        x_square_sum = x_square_sum + first_order_fm
        return [base_out, x_sum, x_square_sum]

    def output_num(self):
        """Define the number of tensors returned by basenet"""
        return 3

    def get_config(self):
        config = {
            "dnn_units_size": self._dnn_units_size,
            "dnn_activation": self._dnn_activation,
        }
        base_config = super(DeepFMbase, self).get_config()
        return {**base_config, **config}


class DeepFMfuse(tf.keras.Model):
    def __init__(self, dnn_units_size, dnn_activation="relu", **kwargs):
        super(DeepFMfuse, self).__init__(**kwargs)
        self._dnn_units_size = dnn_units_size
        self._dnn_activation = dnn_activation
        self._dnn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=self._dnn_activation)
                for units in self._dnn_units_size
            ]
            + [tf.keras.layers.Dense(1)]
        )

    def second_order_fm(self, x_sum_list, x_square_sum):
        x_sum = tf.reduce_sum(x_sum_list, axis=1)
        x_sum = tf.reduce_sum(tf.pow(x_sum, 2), axis=1, keepdims=True)
        x_square_sum = tf.reduce_sum(x_square_sum, axis=1)
        interaction = 0.5 * tf.subtract(
            x_sum,
            x_square_sum,
        )
        return interaction

    def call(self, inputs, **kwargs):
        base_out = tf.concat(inputs[::3], axis=1)
        x_sum_list = tf.stack(inputs[1::3], axis=1)
        x_square_sum = tf.stack(inputs[2::3], axis=1)

        outputs = self.second_order_fm(x_sum_list, x_square_sum) + self._dnn(base_out)
        preds = tf.keras.activations.sigmoid(outputs)
        return preds

    def get_config(self):
        config = {
            "dnn_units_size": self._dnn_units_size,
            "dnn_activation": self._dnn_activation,
        }
        base_config = super(DeepFMfuse, self).get_config()
        return {**base_config, **config}
