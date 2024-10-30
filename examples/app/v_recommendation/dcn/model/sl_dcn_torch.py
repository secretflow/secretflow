# Copyright 2023 Ant Group Co., Ltd.
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
from torch import nn as nn
from torch.nn import functional as F

from secretflow.ml.nn.core.torch import BaseModule


class CatEmbeddingSqrt(nn.Module):

    def __init__(self, categories: List[int], d_embed_max=100, d_cat_sum=32):
        """Embedding and stacking layer for alice and bob.
        Args:
            categories: list of number of categories for each feature
            d_embed_max: max embedding size
            d_cat_sum: sum of embedding size
        """
        super().__init__()
        self.categories = categories
        # logger.debug(f"categories: {categories}")
        self.d_embed_list = [min(max(int(x**0.5), 2), d_embed_max) for x in categories]
        # logger.debug(f"d_embed_list: {self.d_embed_list}")
        self.embedding_list = nn.ModuleList(
            [
                nn.Embedding(self.categories[i], self.d_embed_list[i])
                for i in range(len(categories))
            ]
        )
        # logger.debug(f"embedding_list: {self.embedding_list}")
        # logger.debug(f"d_cat_sum in CatEmbeddingSqrt: {d_cat_sum}")
        assert d_cat_sum == sum(self.d_embed_list)
        self.d_cat_sum = d_cat_sum

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_cat: Long tensor of size ``(batch_size, d_in)``
        """
        # logger.debug(f"x_cat: {x_cat}, and shape: {x_cat.size()}")
        x_out = torch.cat(
            [self.embedding_list[i](x_cat[:, i]) for i in range(len(self.categories))],
            dim=1,
        )
        # logger.debug(f"x_out: {x_out}, and shape: {x_out.size()}")
        return x_out


class Deep(nn.Module):
    def __init__(self, d_in: int, d_layers: List[int], dropout: float):
        """Deep Network for each single party.  The deep network is a fully-connected feed-forward neural network.
        Args:
            d_in: input dimension
            d_layers: list of layer dimensions
            dropout: dropout rate
        """
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
        """Cross Network for each single party.  The key idea of our novel cross network is to apply explicit feature crossing in an efficient way.
        Args:
            d_in: input dimension
            n_cross: number of cross
        """
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
        d_cat_sum: int,
        d_embed_max: int = 8,
        n_cross: int = 2,
        mlp_layers: List[int] = [128, 64, 32],
        mlp_dropout: float = 0.25,
    ):
        """A whole model of DCN for each single party.
        Args:
            d_numerical: 数值特征的维度
            categories: 离散特征的每个类别的最大值
            d_cat_sum: 离散特征的向量嵌入后的总维度
            d_embed_max: 每个离散特征的嵌入最大维度
            n_cross: cross数量
            mlp_layers: deep的每层维度
        """
        super(DCNBase, self).__init__()

        if d_numerical is None:
            d_numerical = 0
        if categories is None:
            categories = []

        # embedding
        self.cat_embedding = (
            CatEmbeddingSqrt(categories, d_embed_max, d_cat_sum) if categories else None
        )

        # deep
        self.d_in = d_numerical
        if self.cat_embedding:
            self.d_in += self.cat_embedding.d_cat_sum

        # deep in: d_in out:mlp_layers[-1]
        self.deep = Deep(self.d_in, mlp_layers, mlp_dropout)

        # cross, d_in = d_out
        self.cross = Cross(self.d_in, n_cross)

        # output

    def forward(self, x):
        """
        x_num : numerical features
        x_cat : categorical features
        """
        x_num, x_cat = x
        # logger.debug(
        #     f"x_num: {x_num}, x_cat: {x_cat}, and x_cat size: {x_cat.size()}, x_num size: {x_num.size()}"
        # )
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
        """Fuse alice and bob's output into a final output.
        Args:
            n_classes: number of classes
            total_fuse_dim: total dimension of the input
        """
        super(DCNFuse, self).__init__()
        self.n_classes = n_classes
        self.stack = nn.Linear(total_fuse_dim, n_classes)
        # self.out = nn.Softmax(dim=1)

    def forward(self, x):
        fuse_input = torch.cat(x, dim=1)
        stack_out = self.stack(fuse_input)
        # if self.n_classes == 1:
        #     x_out = stack_out.unsqueeze(-1)
        # x_out = F.sigmoid(stack_out)

        return stack_out
