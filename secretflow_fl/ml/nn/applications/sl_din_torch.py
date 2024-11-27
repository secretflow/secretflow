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

from typing import Dict, List

import torch
import torch.nn as nn
from secretflow_fl.ml.nn.core.torch import BaseModule


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self.epsilon = 1e-9

    def forward(self, x):
        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        x = self.alpha * x.mul(1 - p) + x.mul(p)
        return x


class ActivationUnit(nn.Module):
    def __init__(self, embedding_dim, dropout=0.2, fc_dims=[32, 16]):
        super(ActivationUnit, self).__init__()
        fc_layers = []
        input_dim = embedding_dim * 4

        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim
        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query, user_behavior):
        seq_len = user_behavior.shape[1]
        queries = torch.cat([query] * seq_len, dim=1)
        attn_input = torch.cat(
            [queries, user_behavior, queries - user_behavior, queries * user_behavior],
            dim=-1,
        )
        out = self.fc(attn_input)
        return out


class AttentionPoolingLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionPoolingLayer, self).__init__()
        self.active_unit = ActivationUnit(embedding_dim=embedding_dim)

    def forward(self, query_ad, user_behavior, mask):
        attns = self.active_unit(query_ad, user_behavior)
        attns = attns.mul(mask)
        user_behavior = user_behavior.mul(attns)
        output = user_behavior.sum(dim=1)
        return output


class DINBase(BaseModule):
    def __init__(
        self,
        fea_list: List[str],
        fea_emb_dim: Dict[str, List[int]],
        target_item_fea: str,
        seq_len: Dict[str, int],
        sequence_fea: List[str] = [],
        padding_idx: int = 0,
    ):
        super(DINBase, self).__init__()
        self.fea_list = fea_list
        self.target_item_fea = target_item_fea
        self.padding_idx = padding_idx
        self.sequence_fea = sequence_fea

        self.fea_embedding = nn.ModuleDict()
        self.AttentionActivate = nn.ModuleDict()

        for fea in fea_list:
            if fea not in sequence_fea:
                self.fea_embedding[fea] = nn.Embedding(
                    fea_emb_dim[fea][0], fea_emb_dim[fea][1]
                )
            else:
                self.fea_embedding[fea] = nn.Embedding(
                    fea_emb_dim[target_item_fea][0], fea_emb_dim[target_item_fea][1]
                )
                self.AttentionActivate[fea] = AttentionPoolingLayer(
                    fea_emb_dim[target_item_fea][1]
                )

    def forward(self, inputs):
        out_emb = []

        for idx, fea in enumerate(self.fea_list):
            if fea not in self.sequence_fea:
                fea_emb = torch.squeeze(self.fea_embedding[fea](inputs[idx]), 1)
                out_emb.append(fea_emb)
            else:
                target_item_index = self.fea_list.index(self.target_item_fea)
                fea_input = inputs[idx]
                target_emb = self.fea_embedding[self.target_item_fea](
                    inputs[target_item_index]
                )
                mask = (fea_input == self.padding_idx).unsqueeze(-1).float()
                user_behavior_emb = self.fea_embedding[fea](fea_input)
                attention_output = self.AttentionActivate[fea](
                    target_emb, user_behavior_emb, mask
                )
                out_emb.append(attention_output)

        out_emb = torch.cat(out_emb, dim=1)
        return out_emb

    def output_num(self):
        return 1


class DINFuse(BaseModule):
    def __init__(self, dnn_units_size):
        super(DINFuse, self).__init__()
        layers = []

        for i in range(1, len(dnn_units_size)):
            layers.append(nn.Linear(dnn_units_size[i - 1], dnn_units_size[i]))
            layers.append(Dice())

        layers.append(nn.Linear(dnn_units_size[-1], 2))
        layers.append(nn.Softmax(dim=1))
        self.dense_internal = nn.Sequential(*layers)

    def forward(self, inputs):
        fuse_input = torch.cat(inputs, dim=1)
        outputs = self.dense_internal(fuse_input)
        return outputs
