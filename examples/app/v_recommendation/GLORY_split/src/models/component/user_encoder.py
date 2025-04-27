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
        self.atte = Sequential('x, mask', [
            (MultiHeadAttention(self.news_dim,
                               self.news_dim,
                               self.news_dim,
                               cfg.model.head_num,
                               cfg.model.head_dim), 'x,x,x,mask -> x'),

            (AttentionPooling(cfg.model.head_num * cfg.model.head_dim, cfg.model.attention_hidden_dim), 'x, mask -> x'),
        ])

    def forward(self, clicked_news, clicked_mask=None):
        result = self.atte(clicked_news, clicked_mask)
        return result

