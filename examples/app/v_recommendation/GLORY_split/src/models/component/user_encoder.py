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


class UserEncoder(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        # layers
        self.atte = Sequential(
            'x, mask',
            [
                (
                    MultiHeadAttention(
                        self.news_dim,
                        self.news_dim,
                        self.news_dim,
                        cfg.model.head_num,
                        cfg.model.head_dim,
                    ),
                    'x,x,x,mask -> x',
                ),
                (
                    AttentionPooling(
                        cfg.model.head_num * cfg.model.head_dim,
                        cfg.model.attention_hidden_dim,
                    ),
                    'x, mask -> x',
                ),
            ],
        )

    def forward(self, clicked_news, clicked_mask=None):
        result = self.atte(clicked_news, clicked_mask)
        return result
