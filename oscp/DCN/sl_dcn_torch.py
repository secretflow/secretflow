import logging
from typing import Dict, List

import torch
from torch import nn as nn
from torch.nn import functional as F

import secretflow as sf
from secretflow.ml.nn.utils import BaseModule


class CatEmbeddingSqrt(nn.Module):
    """
    离散特征使用Embedding层编码, d_embed等于sqrt(category)
    输入shape: [batch_size,d_in],
    输出shape: [batch_size,d_out]
    """

    def __init__(self, categories: List[int], d_embed_list: List[int]):
        super().__init__()
        self.d_embed_list = d_embed_list
        self.categories = categories
        self.embedding_list = nn.ModuleList(
            [
                nn.Embedding(self.categories[i], self.d_embed_list[i])
                for i in range(len(d_embed_list))
            ]
        )

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        param x_cat: Long tensor of size ``(batch_size, d_in)``
        """
        x_out = torch.cat(
            [
                self.embedding_list[i](x_cat[:, i])
                for i in range(len(self.d_embed_list))
            ],
            dim=1,
        )
        return x_out


class Deep(nn.Module):
    def __init__(self, d_in: int, d_layers: List[int], dropout: float):
        super().__init__()
        layers = []
        for d in d_layers:
            layers.append(nn.Linear(d_in, d))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            d_in = d
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Cross(nn.Module):
    def __init__(self, d_in: int, n_cross: int = 2):
        super().__init__()
        self.n_cross = n_cross
        self.linears = nn.ModuleList(
            [nn.Linear(d_in, 1, bias=False) for i in range(self.n_cross)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(d_in)) for i in range(self.n_cross)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xi = x
        for i in range(self.n_cross):
            xi = x0 * self.linears[i](xi) + self.biases[i] + xi
        return xi


class DCNBase(BaseModule):

    def __init__(
        self,
        d_numerical: int,
        d_embed_dict: Dict,
        d_embed_max: int = 8,
        d_in: int = 256,
        n_cross: int = 2,
        mlp_layers: List[int] = [128, 64, 32],
        mlp_dropout: float = 0.25,
    ):
        super(DCNBase, self).__init__()
        '''
        d_numerical: 数值特征的维度
        categories: 离散特征的类别数以及每个类别的最大值
        d_embed_max: 离散特征的embedding维度上限
        '''
        d_embed_list = d_embed_dict.get("d_embed_list")
        categories = d_embed_dict.get("categorical")
        assert len(categories) == len(d_embed_list)

        if d_numerical is None:
            d_numerical = 0
        if d_embed_list is None:
            d_embed_list = []

        self.d_embed_list = d_embed_list

        # embedding
        self.cat_embedding = (
            CatEmbeddingSqrt(categories, d_embed_list) if d_embed_list else None
        )

        # deep
        self.d_in = d_numerical
        if self.cat_embedding:
            self.d_in = d_in

        self.deep = Deep(self.d_in, mlp_layers, mlp_dropout)

        # cross
        self.cross = Cross(self.d_in, n_cross)

        # output

    def forward(self, x):
        # print(x.size())
        """
        x_num : numerical features
        x_cat : categorical features
        """
        x_num, x_cat = x
        x_total = []
        if x_num is not None:
            x_total.append(x_num)
        if self.cat_embedding is not None:
            x_total.append(self.cat_embedding(x_cat))

        x_total = torch.cat(x_total, dim=-1)

        # cross
        x_cross = self.cross(x_total)

        # deep
        x_deep = self.deep(x_total)

        # output
        x_cross_deep = torch.cat([x_cross, x_deep], dim=-1)
        return x_cross_deep

    def output_num(self):
        """Define the number of tensors returned by basenet"""
        return 1


class DCNFuse(BaseModule):

    def __init__(self, n_classes=2, total_fuse_dim=32):
        super(DCNFuse, self).__init__()
        self.n_classes = n_classes
        self.stack = nn.Linear(total_fuse_dim, n_classes)
        # self.out = nn.Softmax(dim=1)

    def forward(self, x):
        fuse_input = torch.cat(x, dim=1)
        stack_out = self.stack(fuse_input)
        # if self.n_classes == 1:
        #     x_out = stack_out.squeeze(-1)
        x_out = F.sigmoid(stack_out)

        return x_out
