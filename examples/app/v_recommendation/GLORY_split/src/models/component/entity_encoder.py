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

import torch
import torch.nn as nn
from models.base.layers import *
from torch_geometric.nn import Sequential


class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = 400

        self.atte = Sequential(
            'x, mask',
            [
                (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
                (
                    MultiHeadAttention(
                        self.entity_dim,
                        self.entity_dim,
                        self.entity_dim,
                        int(self.entity_dim / cfg.model.head_dim),
                        cfg.model.head_dim,
                    ),
                    'x,x,x,mask -> x',
                ),
                nn.LayerNorm(self.entity_dim),
                nn.Dropout(p=cfg.dropout_probability),
                (
                    AttentionPooling(self.entity_dim, cfg.model.attention_hidden_dim),
                    'x, mask-> x',
                ),
                nn.LayerNorm(self.entity_dim),
                nn.Linear(self.entity_dim, self.news_dim),
                nn.LeakyReLU(0.2),
            ],
        )

    def forward(self, entity_input, entity_mask=None):

        batch_size, num_news, num_entity, _ = entity_input.shape

        if entity_mask is not None:
            result = self.atte(
                entity_input.view(batch_size * num_news, num_entity, self.entity_dim),
                entity_mask.view(batch_size * num_news, num_entity),
            ).view(batch_size, num_news, self.news_dim)
        else:
            result = self.atte(
                entity_input.view(batch_size * num_news, num_entity, self.entity_dim),
                None,
            ).view(batch_size, num_news, self.news_dim)

        return result


class GlobalEntityEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.entity_dim = cfg.model.entity_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim

        self.atte = Sequential(
            'x, mask',
            [
                (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
                (
                    MultiHeadAttention(
                        self.entity_dim,
                        self.entity_dim,
                        self.entity_dim,
                        cfg.model.head_num,
                        cfg.model.head_dim,
                    ),
                    'x,x,x,mask -> x',
                ),
                nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
                nn.Dropout(p=cfg.dropout_probability),
                (
                    AttentionPooling(
                        cfg.model.head_num * cfg.model.head_dim,
                        cfg.model.attention_hidden_dim,
                    ),
                    'x, mask-> x',
                ),
                nn.LayerNorm(cfg.model.head_num * cfg.model.head_dim),
            ],
        )

    def forward(self, entity_input, entity_mask=None):

        batch_size, num_news, num_entity, _ = entity_input.shape
        if entity_mask is not None:
            entity_mask = entity_mask.view(batch_size * num_news, num_entity)

        result = self.atte(
            entity_input.view(batch_size * num_news, num_entity, self.entity_dim),
            entity_mask,
        ).view(batch_size, num_news, self.news_dim)

        return result
