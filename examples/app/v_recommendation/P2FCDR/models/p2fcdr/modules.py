# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from . import config


class MLP(nn.Module):
    """MLP module.
    """

    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.layer_number = config.num_mlp_layers
        self.encoder = []
        for i in range(self.layer_number):
            num_features = (config.emb_size if i == 0 else config.hidden_size)
            num_hidden = (config.emb_size
                          if i == self.layer_number - 1
                          else config.hidden_size)
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
            num_features = (config.emb_size if i == 0 else config.hidden_size)
            num_hidden = (config.emb_size
                          if i == self.layer_number - 1
                          else config.hidden_size)
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
