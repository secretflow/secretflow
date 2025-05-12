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


class ClickEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.news_dim = 400
        self.use_entity = cfg.model.use_entity
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
                ],
            )
        else:
            self.atte = Sequential(
                'a,b',
                [
                    (
                        lambda a, b: torch.stack([a, b], dim=-2).view(
                            -1, 2, self.news_dim
                        ),
                        'a,b -> x',
                    ),
                    AttentionPooling(self.news_dim, cfg.model.attention_hidden_dim),
                ],
            )

    def forward(self, clicke_title_emb, click_graph_emb, click_entity_emb=None):

        batch_size, num_news = clicke_title_emb.shape[0], clicke_title_emb.shape[1]
        if click_entity_emb is not None:
            result = self.atte(clicke_title_emb, click_graph_emb, click_entity_emb)
        else:
            result = self.atte(clicke_title_emb, click_graph_emb)

        return result.view(batch_size, num_news, self.news_dim)
