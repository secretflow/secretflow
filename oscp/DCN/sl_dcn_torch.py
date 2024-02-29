import logging
from typing import Dict, List

import torch
from torch import nn as nn
from torch.nn import functional as F

from secretflow.ml.nn.utils import BaseModule

fm = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s] [%(funcName)s:%(lineno)d]"
# 设置日志级别 打印日志
logging.basicConfig(level=logging.DEBUG, format=fm, filename="testlog//log01.log")
# 基本用法


class CatEmbeddingSqrt(nn.Module):
    """
    离散特征使用Embedding层编码, d_embed等于sqrt(category)
    输入shape: [batch_size,d_in],
    输出shape: [batch_size,d_out]
    """

    def __init__(self, categories: List[int], d_embed_max: int = 100):
        super().__init__()
        self.categories = categories
        self.d_embed_list = [min(max(int(x**0.5), 2), d_embed_max) for x in categories]
        self.embedding_list = nn.ModuleList(
            [
                nn.Embedding(self.categories[i], self.d_embed_list[i])
                for i in range(len(categories))
            ]
        )
        self.d_cat_sum = sum(self.d_embed_list)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        param x_cat: Long tensor of size ``(batch_size, d_in)``
        """
        x_out = torch.cat(
            [self.embedding_list[i](x_cat[:, i]) for i in range(len(self.categories))],
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
        categories: List[int],
        d_embed_max: int = 8,
        n_cross: int = 2,
        mlp_layers: List[int] = [128, 64, 32],
        mlp_dropout: float = 0.25,
    ):
        super(DCNBase, self).__init__()
        '''
        d_numerical: 数值特征的维度
        categories: 离散特征的类别数
        d_embed_max: 离散特征的embedding维度上限
        '''

        if d_numerical is None:
            d_numerical = 0
        if categories is None:
            categories = []

        self.categories = categories

        # embedding
        self.cat_embedding = (
            CatEmbeddingSqrt(categories, d_embed_max) if categories else None
        )

        # deep
        self.d_in = d_numerical
        if self.cat_embedding:
            self.d_in += self.cat_embedding.d_cat_sum

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
        logging.debug("x:{}".format(x))

        x_num, x_cat = x
        logging.debug(
            "x_num:{}".format(x_num),
            "x_cat:{}".format(x_cat),
            "x_num 's size:{}".format(x_num.size()),
            "x_cat 's size:{}".format(x_cat.size()),
        )
        # embedding
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


class DCNFuse(BaseModule):
    def __init__(self, n_classes=2, deep_dim_out=9, cross_dim_out=32):
        super(DCNFuse, self).__init__()

        self.stack = nn.Linear(2 * (deep_dim_out + cross_dim_out), n_classes)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        fuse_input = torch.cat(x, dim=1)

        return self.out(self.stack(fuse_input))
