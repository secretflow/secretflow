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

from typing import Dict, List

import tensorflow as tf
from tensorflow.keras import layers


class BSTBase(tf.keras.Model):
    def __init__(
        self,
        preprocess_layer: tf.keras.Model,
        sequence_fea: List[str] = [],
        item_embedding_dims: Dict[str, int] = {},
        seq_len: Dict[str, int] = {},
        item_voc_size: Dict[str, int] = {},
        num_head: Dict[str, int] = {},
        dropout_rate: Dict[str, float] = {},
        **kwargs,
    ):
        """Split learning version of BST
        Args:
            preprocess_layer: the preprocessed layer a keras model, output a dict of preprocessed data
            sequence_fea: list[string] or empty list, list of sequence feature name
            item_embedding_dims: dict[string, int] or empty dict, feature name and its embedding dim for each sequence feature, every element in sequence_fea should be found in this dict
            seq_len: dict[string, int] or empty dict, feature name and its embedding dim for each sequence feature, every element in sequence_fea should be found in this dict
            item_voc_size: dict[string, int] or empty dict, feature name and its vocabulary size for each sequence feature, every element in sequence_fea should be found in this dict
            num_head: dict[string, int] or empty dict, feature name and its head number in attention layer for each sequence feature, every element in sequence_fea should be found in this dict
            dropout_rate: dict[string, float] or empty dict, feature name and its dropout rate for each sequence feature, every element in sequence_fea should be found in this dict
        """

        super(BSTBase, self).__init__(**kwargs)

        self.preprocess = preprocess_layer
        self.sequence_fea = sequence_fea

        self._seq_len = 0
        # item embedding
        if len(sequence_fea) > 0:
            self._seq_len = seq_len
            self._item_embedding_encoder = {}
            self._position_embedding_encoder = {}
            self._attention_layer = {}
            self._normalize_layer = {}
            self._dense_layer = {}
            self._dropout_layer = {}

            for fea in sequence_fea:
                self._item_embedding_encoder[fea] = layers.Embedding(
                    input_dim=item_voc_size[fea] + 2,
                    output_dim=item_embedding_dims[fea],
                    mask_zero=True,
                    name=f"{fea}_item_embedding",
                )

                # position embedding
                self._position_embedding_encoder[fea] = layers.Embedding(
                    input_dim=seq_len[fea] + 1,
                    output_dim=item_embedding_dims[fea],
                    name=f"{fea}_position_embedding",
                )

                self._attention_layer[fea] = layers.MultiHeadAttention(
                    name=f"{fea}_mhsa",
                    num_heads=num_head[fea],
                    key_dim=item_embedding_dims[fea],
                    dropout=dropout_rate[fea],
                )
                self._normalize_layer[fea] = layers.LayerNormalization()
                self._dense_layer[fea] = layers.Dense(units=item_embedding_dims[fea])
                self._dropout_layer[fea] = layers.Dropout(dropout_rate[fea])

    def call(self, inputs, **kwargs):
        preprocess_data = self.preprocess(inputs, training=True)

        encoded_fea = []
        for key in preprocess_data:
            if key not in self.sequence_fea:
                emb_sqz = tf.squeeze(preprocess_data[key], [1])
                encoded_fea.append(emb_sqz)
            else:
                mask = self._item_embedding_encoder[key].compute_mask(
                    preprocess_data[key]
                )
                encoded_sequence_items = self._item_embedding_encoder[key](
                    preprocess_data[key]
                )

                positions = tf.range(start=0, limit=self._seq_len[key], delta=1)
                encodded_positions = self._position_embedding_encoder[key](positions)

                encoded_sequence_items = encoded_sequence_items + encodded_positions

                mask_expand = mask[:, tf.newaxis, tf.newaxis, :]
                attention_output = self._attention_layer[key](
                    encoded_sequence_items,
                    encoded_sequence_items,
                    attention_mask=mask_expand,
                )

                attention_output = layers.Add()(
                    [encoded_sequence_items, attention_output]
                )
                # make sequence fixed length, avg pooling, [bs, emb_size]
                mask = tf.cast(mask, tf.float32)
                mask = tf.stop_gradient(mask)
                mask_sum = tf.reduce_sum(mask, axis=-1, keepdims=True)
                mask = tf.divide(mask, mask_sum)
                mask = mask[:, :, tf.newaxis]
                masked_att_output = layers.Multiply(name="mul_mask")(
                    [attention_output, mask]
                )

                masked_att_output = tf.reduce_sum(masked_att_output, axis=-2)

                x1 = self._dropout_layer[key](masked_att_output)
                x1 = self._normalize_layer[key](x1)
                x2 = layers.LeakyReLU()(x1)
                x2 = self._dense_layer[key](x2)
                x2 = self._dropout_layer[key](x2)
                transformer_features = layers.Add()([x1, x2])
                transformer_features = self._normalize_layer[key](transformer_features)
                features = layers.Flatten()(transformer_features)

                encoded_fea.append(features)

        if len(encoded_fea) == 1:
            return encoded_fea
        else:
            out = tf.concat(encoded_fea, axis=-1)
            return [out]

    def output_num(self):
        """Define the number of tensors returned by basenet"""
        return 1

    def get_config(self):
        config = {
            "seq_len": self._seq_len,
        }
        base_config = super(BSTBase, self).get_config()
        return {**base_config, **config}


class AttentionSequencePoolingLayer(layers.Layer):
    def __init__(
        self,
        att_hidden_units=(8, 4),
        att_activation='sigmoid',
        weight_normalization=True,
        return_score=False,
        **kwargs,
    ):
        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)

        self.dense_internal = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units, activation=self.att_activation)
                for units in self.att_hidden_units
            ]
            + [tf.keras.layers.Dense(1)]
        )

    def build(self, input_shape):
        super(AttentionSequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        queries, keys, key_masks = inputs
        keys_len = keys.get_shape()[1]
        queries = tf.keras.backend.repeat_elements(queries, keys_len, 1)

        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        att_out = self.dense_internal(att_input)

        outputs = tf.transpose(att_out, (0, 2, 1))

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-(2**32) + 1)
        else:
            paddings = tf.zeros_like(outputs)

        outputs = tf.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = tf.nn.softmax(outputs)

        if not self.return_score:
            outputs = tf.matmul(outputs, keys)

        return outputs


class BSTPlusBase(tf.keras.Model):
    def __init__(
        self,
        preprocess_layer: tf.keras.Model,
        sequence_fea: List[str] = [],
        target_fea: str = None,
        item_embedding_dims: Dict[str, int] = {},
        seq_len: Dict[str, int] = {},
        item_voc_size: Dict[str, int] = {},
        num_head: Dict[str, int] = {},
        dropout_rate: Dict[str, float] = {},
        use_positional_encoding=True,
        use_res=True,
        use_feed_forward=True,
        use_layer_norm=False,
        **kwargs,
    ):
        """Split learning version of BST
        Args:
            preprocess_layer: the preprocessed layer a keras model, output a dict of preprocessed data
            sequence_fea: list[string] or empty list, list of sequence feature name
            item_embedding_dims: dict[string, int] or empty dict, feature name and its embedding dim for each sequence feature, every element in sequence_fea should be found in this dict
            seq_len: dict[string, int] or empty dict, feature name and its embedding dim for each sequence feature, every element in sequence_fea should be found in this dict
            item_voc_size: dict[string, int] or empty dict, feature name and its vocabulary size for each sequence feature, every element in sequence_fea should be found in this dict
            num_head: dict[string, int] or empty dict, feature name and its head number in attention layer for each sequence feature, every element in sequence_fea should be found in this dict
            dropout_rate: dict[string, int] or empty dict, feature name and its dropout rate for each sequence feature, every element in sequence_fea should be found in this dict
        """

        super(BSTPlusBase, self).__init__(**kwargs)

        self.preprocess = preprocess_layer
        self.sequence_fea = sequence_fea
        self.target_fea = None

        self.use_positional_encoding = use_positional_encoding
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.use_layer_norm = use_layer_norm

        self._seq_len = 0
        # item embedding
        if len(sequence_fea) > 0:
            self._seq_len = seq_len
            self._position_embedding_encoder = {}
            self._attention_layer = {}
            self._normalize_layer = {}
            self._dense_layer1 = {}
            self._dense_layer2 = {}
            self._dropout_layer = {}
            self._target_attention = {}

            assert target_fea is not None
            self.target_fea = target_fea
            # we share lookup table for history sequences and target item
            self._item_embedding_encoder = layers.Embedding(
                input_dim=item_voc_size[self.target_fea] + 2,
                output_dim=item_embedding_dims[self.target_fea],
                mask_zero=True,
                name=f"target_item_embedding",
            )

            item_emb_dim = item_embedding_dims[self.target_fea]
            for fea in sequence_fea:
                self._attention_layer[fea] = layers.MultiHeadAttention(
                    name=f"{fea}_mhsa",
                    num_heads=num_head[fea],
                    key_dim=item_emb_dim,
                    dropout=dropout_rate[fea],
                )

                if self.use_positional_encoding:
                    # position embedding
                    self._position_embedding_encoder[fea] = layers.Embedding(
                        input_dim=seq_len[fea] + 1,
                        output_dim=item_emb_dim,
                        name=f"{fea}_position_embedding",
                    )

                if self.use_layer_norm:
                    self._normalize_layer[fea] = layers.LayerNormalization()
                if self.use_feed_forward:
                    self._dense_layer1[fea] = layers.Dense(
                        units=item_emb_dim, activation="relu"
                    )
                    self._dense_layer2[fea] = layers.Dense(units=item_emb_dim)

                self._dropout_layer[fea] = layers.Dropout(dropout_rate[fea])

                self._target_attention[fea] = AttentionSequencePoolingLayer()

    def call(self, inputs, **kwargs):
        preprocess_data = self.preprocess(inputs, training=True)
        if self.target_fea is not None:
            encoded_target = self._item_embedding_encoder(
                preprocess_data[self.target_fea]
            )

        encoded_fea = []
        for key in preprocess_data:
            if key not in self.sequence_fea:
                if key != self.target_fea:
                    emb_sqz = tf.squeeze(preprocess_data[key], [1])
                    encoded_fea.append(emb_sqz)
            else:
                # prepare inputs for mhsa
                mask = self._item_embedding_encoder.compute_mask(preprocess_data[key])
                mask = tf.stop_gradient(mask)  # maybe no need
                encoded_sequence_items = self._item_embedding_encoder(
                    preprocess_data[key]
                )

                if self.use_positional_encoding:
                    positions = tf.range(start=0, limit=self._seq_len[key], delta=1)
                    encodded_positions = self._position_embedding_encoder[key](
                        positions
                    )

                    encoded_sequence_items = encoded_sequence_items + encodded_positions

                mask_expand = mask[:, tf.newaxis, tf.newaxis, :]
                # mhsa
                attention_output = self._attention_layer[key](
                    encoded_sequence_items,
                    encoded_sequence_items,
                    attention_mask=mask_expand,
                )

                # postprocess for mhsa
                if self.use_res:
                    attention_output = layers.Add()(
                        [encoded_sequence_items, attention_output]
                    )
                if self.use_layer_norm:
                    attention_output = self._normalize_layer[key](attention_output)
                if self.use_feed_forward:
                    attention_output_ori = attention_output
                    x1 = self._dense_layer1[key](attention_output)
                    x1 = self._dropout_layer[key](x1)
                    attention_output = self._dense_layer2[key](x1)
                    if self.use_res:
                        attention_output = layers.Add()(
                            [attention_output, attention_output_ori]
                        )
                    if self.use_layer_norm:
                        attention_output = layers.LayerNormalization()(attention_output)

                # target attention
                # as attention_mask in mhsa is key_mask, mhsa does not take query_mask into account
                # we must mask here(mhsa' query mask => att_seq_pooling's key mask)
                mask = mask[:, tf.newaxis, :]
                target_output = self._target_attention[key](
                    [encoded_target, attention_output, mask]
                )
                print('target_output.shape: ', target_output.shape)
                target_output = tf.squeeze(target_output, axis=[1])

                encoded_fea.append(target_output)

        if len(encoded_fea) == 1:
            return encoded_fea
        else:
            out = tf.concat(encoded_fea, axis=-1)
            return [out]

    def output_num(self):
        """Define the number of tensors returned by basenet"""
        return 1

    def get_config(self):
        config = {
            "seq_len": self._seq_len,
        }
        base_config = super(BSTPlusBase, self).get_config()
        return {**base_config, **config}


class BSTFuse(tf.keras.Model):
    def __init__(self, dnn_units_size, dnn_activation="relu", **kwargs):
        super(BSTFuse, self).__init__(**kwargs)
        self._dnn_units_size = dnn_units_size
        self._dnn_activation = dnn_activation
        self._dnn = tf.keras.Sequential(
            [
                layers.Dense(units, activation=self._dnn_activation)
                for units in self._dnn_units_size
            ]
            + [layers.Dense(1)]
        )

    def call(self, inputs, **kwargs):
        fuse_input = tf.concat(inputs, axis=1)
        outputs = self._dnn(fuse_input)
        preds = tf.keras.activations.sigmoid(outputs)
        return preds

    def get_config(self):
        config = {
            "dnn_units_size": self._dnn_units_size,
            "dnn_activation": self._dnn_activation,
        }
        base_config = super(BSTFuse, self).get_config()
        return {**base_config, **config}
