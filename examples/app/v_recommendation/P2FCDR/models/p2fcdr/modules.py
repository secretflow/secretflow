# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.functional as F
from . import config


class MLP(nn.Module):
    """MLP module."""

    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.layer_number = config.num_mlp_layers
        self.encoder = []
        for i in range(self.layer_number):
            num_features = config.emb_size if i == 0 else config.hidden_size
            num_hidden = (
                config.emb_size if i == self.layer_number - 1 else config.hidden_size
            )
            if i == 0:
                self.encoder.append(nn.Linear(num_features * 2, num_hidden))
            else:
                self.encoder.append(nn.Linear(num_features, num_hidden))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = config.dropout_rate

    def forward(self, input):
        for layer in self.encoder:
            # input= F.dropout(
            #     input, self.dropout, training=self.training)
            input = layer(input)
            input = F.relu(input)
        return input


class GateMLP(nn.Module):
    """MLP module utilized to automatically learn and derive
    the gated selecting vector.
    """

    def __init__(self, args):
        super(GateMLP, self).__init__()
        self.args = args
        self.layer_number = config.num_gmlp_layers
        self.encoder = []
        for i in range(self.layer_number):
            num_features = config.emb_size if i == 0 else config.hidden_size
            num_hidden = (
                config.emb_size if i == self.layer_number - 1 else config.hidden_size
            )
            self.encoder.append(nn.Linear(num_features, num_hidden))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = config.dropout_rate

    def forward(self, input):
        for layer in self.encoder[:-1]:
            # input= F.dropout(
            #     input, self.dropout, training=self.training)
            input = layer(input)
            input = F.relu(input)
        input = self.encoder[-1](input)
        input = F.sigmoid(input)

        return input
