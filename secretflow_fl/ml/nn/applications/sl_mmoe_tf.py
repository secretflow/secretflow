# Copyright 2023 Ant Group Co., Ltd.
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

from typing import List

import tensorflow as tf


class MMoEBase(tf.keras.Model):
    def __init__(
        self,
        dnn_units_size: List[int],
        dnn_activation: str = "relu",
        preprocess_layer: tf.keras.Model = None,
        embedding_dim: int = 8,
        **kwargs,
    ):
        """Split learning version of MMoe
        Args:
            dnn_units_size: list,list of positive integer or empty list, the layer number and units in each layer of DNN
            dnn_activation: activation function of dnn part
            preprocess_layer: The preprocessed layer a keras model, output a dict of preprocessed data
            embedding_dim: embedding dim, default to be 8
        """

        super(MMoEBase, self).__init__(**kwargs)

        self.preprocess = preprocess_layer
        outputs = self.preprocess.output

        self._embedding_features_layer = {
            k: tf.keras.layers.Dense(units=embedding_dim) for k, v in outputs.items()
        }

        self._dnn_units_size = dnn_units_size
        self._dnn_activation = dnn_activation

        if len(dnn_units_size) > 0:
            self.dense_internal = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(units, activation=self._dnn_activation)
                    for units in self._dnn_units_size
                ]
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

        base_out = tf.concat(embedding_features, axis=-1)
        if len(self._dnn_units_size) > 0:
            base_out = self.dense_internal(base_out)

        return [base_out]

    def output_num(self):
        """Define the number of tensors returned by basenet"""
        return 1

    def get_config(self):
        config = {
            "dnn_units_size": self._dnn_units_size,
            "dnn_activation": self._dnn_activation,
        }
        base_config = super(MMoEBase, self).get_config()
        return {**base_config, **config}


class MMoEFuse(tf.keras.Model):
    def __init__(
        self,
        num_experts: int,
        expert_units_size: List[int],
        expert_activation: str,
        num_tasks: int,
        gate_units_size: List[int],
        gate_activation: str,
        tower_units_size: List[int],
        tower_activation: str,
        output_activation: List[str],
        **kwargs,
    ):
        super(MMoEFuse, self).__init__(**kwargs)
        self._num_experts = num_experts
        self._num_tasks = num_tasks
        self._expert_units_size = expert_units_size

        self._expert_layers = []
        self._gate_layers = []
        self._tower_layers = []

        for i in range(self._num_experts):
            self._expert_layers.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(units, activation=expert_activation)
                        for units in expert_units_size
                    ]
                )
            )

        for i in range(self._num_tasks):
            self._gate_layers.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(units, activation=gate_activation)
                        for units in gate_units_size
                    ]
                    + [tf.keras.layers.Dense(self._num_experts, activation="softmax")]
                )
            )

            self._tower_layers.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(units, activation=tower_activation)
                        for units in tower_units_size
                    ]
                    + [tf.keras.layers.Dense(1, activation=output_activation[i])]
                )
            )

    def call(self, inputs, **kwargs):
        fuse_input = tf.concat(inputs, axis=1)
        expert_outputs = []
        for expert_layer in self._expert_layers:
            # [bs, hidden_size, 1]
            expert_output = tf.expand_dims(expert_layer(fuse_input), axis=2)
            expert_outputs.append(expert_output)
        expert_outputs = tf.concat(expert_outputs, 2)  # [bs, hidden_size, num_experts]

        gate_outputs = []
        for gate_layer in self._gate_layers:
            gate_outputs.append(gate_layer(fuse_input))  # num_tasks * [bs, num_experts]

        final_outputs = []
        for i in range(len(gate_outputs)):
            expanded_gate_output = tf.expand_dims(
                gate_outputs[i], axis=1
            )  # [bs, 1, num_experts]
            # [bs, hidden_size, num_experts]
            weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(
                expanded_gate_output, self._expert_units_size[-1], axis=1
            )
            tower_input = tf.reduce_sum(weighted_expert_output, 2)

            final_outputs.append(self._tower_layers[i](tower_input))

        return final_outputs

    def get_config(self):
        config = {
            "num_experts": self._num_experts,
            "num_tasks": self._num_tasks,
            "expert_units_size": self._expert_units_size,
        }
        base_config = super(MMoEFuse, self).get_config()
        return {**base_config, **config}
