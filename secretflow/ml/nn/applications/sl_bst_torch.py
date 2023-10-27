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

from typing import List, Dict

import torch
from torch import nn as nn
from torch.nn import functional as F

from secretflow.ml.nn.utils import BaseModule


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class BSTBase(BaseModule):
    def __init__(
        self,
        fea_list: List[str],
        fea_emb_dim: Dict[str, List[int]],
        target_item_fea: str,
        seq_len: Dict[str, int],
        sequence_fea: List[str] = [],
        padding_idx: int = 0,
    ):
        """Split learning version of BST
        Args:
            fea_list: list[string], all feature names containing sequence features
            fea_emb_dim: dict[string, list[int]], key is feature_name, value[0] is feature's input dimension size, value[1] is feature's output dimension size
            sequence_fea: list[string] or empty list, list of sequence feature name
            target_item_fea: target item feature name, it will share embedding lookup table with sequence features
            seq_len: dict[string, int] or empty dict, feature name and its embedding dim for each sequence feature, every element in sequence_fea should be found in this dict
            padding_idx: sequence feature's padding idx
        """

        super(BSTBase, self).__init__()

        self.target_item_fea = target_item_fea
        self.fea_embedding = {}
        self.fea_list = fea_list
        for fea in fea_list:
            if fea not in sequence_fea:
                self.fea_embedding[fea] = nn.Embedding(
                    fea_emb_dim[fea][0], fea_emb_dim[fea][1]
                )

        self.fea_embedding = nn.ModuleDict(self.fea_embedding)

        self.padding_idx = padding_idx
        self.sequence_fea = sequence_fea
        self.positional_embedding = {}
        self.transfomerlayer = {}
        for fea in sequence_fea:
            self.positional_embedding[fea] = PositionalEmbedding(
                seq_len[fea], fea_emb_dim[self.target_item_fea][1]
            )
            self.transfomerlayer[fea] = nn.TransformerEncoderLayer(
                fea_emb_dim[self.target_item_fea][1],
                nhead=3,
                dim_feedforward=fea_emb_dim[self.target_item_fea][1],
                dropout=0.2,
            )

        self.positional_embedding = nn.ModuleDict(self.positional_embedding)
        self.transfomerlayer = nn.ModuleDict(self.transfomerlayer)

    def forward(self, inputs):
        out_emb = []
        for idx, fea in enumerate(self.fea_list):
            if fea not in self.sequence_fea:
                fea_emb = torch.squeeze(self.fea_embedding[fea](inputs[idx]), 1)
                out_emb.append(fea_emb)
            else:
                fea_input = inputs[idx]
                mask = fea_input == self.padding_idx
                fea_emb = self.fea_embedding[self.target_item_fea](fea_input)

                pos_emb = self.positional_embedding[fea](fea_emb)
                transfomer_features = pos_emb + fea_emb

                # do not mask query also
                transformer_output = self.transfomerlayer[fea](
                    transfomer_features.transpose(0, 1), src_key_padding_mask=mask
                ).transpose(0, 1)
                # make sure [PAD],[PAD],[PAD],[PAD] not in input, though we masked_fill inf here, the model will diverge
                transformer_output = transformer_output.masked_fill(
                    torch.isnan(transformer_output), 0
                )

                mask = mask.unsqueeze(2).repeat(1, 1, fea_emb.size()[-1])
                paddings = torch.zeros_like(transformer_output)
                transformer_output = torch.where(mask, paddings, transformer_output)

                transformer_output = torch.flatten(transformer_output, start_dim=1)
                out_emb.append(transformer_output)

        out = torch.cat((out_emb), dim=1)
        return out

    def output_num(self):
        """Define the number of tensors returned by basenet"""
        return 1


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(
        self,
        att_hidden_units=(8, 4),
        att_activation='sigmoid',
        weight_normalization=True,
        return_score=False,
        **kwargs
    ):
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score

        layers = []
        for i in range(1, len(self.att_hidden_units)):
            layers.append(
                nn.Linear(self.att_hidden_units[i - 1], self.att_hidden_units[i])
            )
            if att_activation == 'sigmoid':
                layers.append(nn.Sigmoid())
        layers.append(nn.Linear(self.att_hidden_units[-1], 1))
        self.dense_internal = nn.Sequential(*layers)

    def forward(self, inputs):
        queries, keys, key_masks = inputs
        keys_len = keys.size()[1]
        queries = queries.unsqueeze(1).repeat(1, keys_len, 1)

        att_input = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        att_out = self.dense_internal(att_input)

        outputs = att_out.transpose(1, 2)

        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-(2**32) + 1)
        else:
            paddings = torch.zeros_like(outputs)

        outputs = torch.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = F.softmax(outputs, dim=-1)

        if not self.return_score:
            outputs = torch.matmul(outputs, keys)

        return outputs


class BSTBasePlus(BaseModule):
    def __init__(
        self,
        fea_list: List[str],
        fea_emb_dim: Dict[str, List[int]],
        sequence_fea: List[str],
        target_item_fea: str,
        seq_len: Dict[str, int],
        padding_idx: int = 0,
    ):
        """Split learning version of BST, and use AttentionSequencePoolingLayer to process target feature and sequence features
        Args:
            fea_list: list[string], all feature names containing sequence features
            fea_emb_dim: dict[string, list[int]], key is feature_name, value[0] is feature's input dimension size, value[1] is feature's output dimension size
            sequence_fea: list[string] or empty list, list of sequence feature name
            target_item_fea: target item feature name, it will share embedding lookup table with sequence features
            seq_len: dict[string, int] or empty dict, feature name and its embedding dim for each sequence feature, every element in sequence_fea should be found in this dict
            padding_idx: sequence feature's padding idx
        """

        super(BSTBasePlus, self).__init__()

        self.target_item_fea = target_item_fea
        self.fea_embedding = {}
        self.fea_list = fea_list
        for fea in fea_list:
            if fea not in sequence_fea:
                self.fea_embedding[fea] = nn.Embedding(
                    fea_emb_dim[fea][0], fea_emb_dim[fea][1]
                )

        self.fea_embedding = nn.ModuleDict(self.fea_embedding)

        self.padding_idx = padding_idx
        self.sequence_fea = sequence_fea
        self.positional_embedding = {}
        self.transfomerlayer = {}
        self.att_pooling = {}
        for fea in sequence_fea:
            self.positional_embedding[fea] = PositionalEmbedding(
                seq_len[fea], fea_emb_dim[self.target_item_fea][1]
            )
            self.transfomerlayer[fea] = nn.TransformerEncoderLayer(
                fea_emb_dim[self.target_item_fea][1],
                nhead=3,
                dim_feedforward=fea_emb_dim[self.target_item_fea][1],
                dropout=0.2,
            )
            self.att_pooling[fea] = AttentionSequencePoolingLayer(
                att_hidden_units=[36, 16]
            )

        self.positional_embedding = nn.ModuleDict(self.positional_embedding)
        self.transfomerlayer = nn.ModuleDict(self.transfomerlayer)
        self.att_pooling = nn.ModuleDict(self.att_pooling)

    def forward(self, inputs):
        out_emb = []
        target_emb = None
        for idx, fea in enumerate(self.fea_list):
            if fea not in self.sequence_fea:
                fea_emb = torch.squeeze(self.fea_embedding[fea](inputs[idx]), 1)
                out_emb.append(fea_emb)
                if fea == self.target_item_fea:
                    target_emb = fea_emb

        for idx, fea in enumerate(self.fea_list):
            if fea in self.sequence_fea:
                mask = inputs[idx] == self.padding_idx
                fea_emb = self.fea_embedding[self.target_item_fea](inputs[idx])

                pos_emb = self.positional_embedding[fea](fea_emb)
                transfomer_features = pos_emb + fea_emb
                transformer_output = self.transfomerlayer[fea](
                    transfomer_features.transpose(0, 1), src_key_padding_mask=mask
                ).transpose(0, 1)

                transformer_output = transformer_output.masked_fill(
                    torch.isnan(transformer_output), 0
                )

                mask = mask.unsqueeze(1)
                att_out = self.att_pooling[fea]([target_emb, transformer_output, mask])
                att_out = torch.squeeze(att_out, 1)
                out_emb.append(att_out)

        out = torch.cat((out_emb), dim=1)
        return out

    def output_num(self):
        """Define the number of tensors returned by basenet"""
        return 1


class BSTfuse(BaseModule):
    def __init__(self, dnn_units_size, dnn_activation="relu", **kwargs):
        super(BSTfuse, self).__init__()
        layers = []
        for i in range(1, len(dnn_units_size)):
            layers.append(nn.Linear(dnn_units_size[i - 1], dnn_units_size[i]))
            if dnn_activation == 'relu':
                layers.append(nn.ReLU())
        layers.append(nn.Linear(dnn_units_size[-1], 2))
        layers.append(nn.Softmax(dim=1))
        self.dense_internal = nn.Sequential(*layers)

    def forward(self, inputs):
        fuse_input = torch.cat(inputs, dim=1)
        outputs = self.dense_internal(fuse_input)
        return outputs
