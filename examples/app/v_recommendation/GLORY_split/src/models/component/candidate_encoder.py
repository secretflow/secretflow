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
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential


class CandidateEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_entity = cfg.model.use_entity

        self.entity_dim = 100
        self.news_dim = cfg.model.head_dim * cfg.model.head_num
        self.output_dim = cfg.model.head_dim * cfg.model.head_num

        if self.use_entity:
            self.atte = Sequential(
                'a,b,c',
                [
                    (
                        lambda a, b, c: torch.stack([a, b, c], dim=-2).view(
                            -1, 3, self.news_dim
                        ),
                        'a,b,c -> x',
                    ),
                    AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                    nn.Linear(self.news_dim, self.output_dim),
                    nn.LeakyReLU(0.2),
                ],
            )
        else:
            self.atte = Sequential(
                'a,b,c',
                [
                    (nn.Linear(self.news_dim, self.output_dim), 'a -> x'),
                    nn.LeakyReLU(0.2),
                ],
            )

    def forward(self, candidate_emb, origin_emb=None, neighbor_emb=None):

        batch_size, num_news = candidate_emb.shape[0], candidate_emb.shape[1]

        result = self.atte(candidate_emb, origin_emb, neighbor_emb).view(
            batch_size, num_news, self.output_dim
        )

        return result
