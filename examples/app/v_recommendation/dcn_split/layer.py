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
import torch
import torch.nn as nn


class Deep(nn.Module):
    def __init__(self, input_dim, deep_layers):
        super(Deep, self).__init__()

        deep_layers.insert(0, input_dim)
        deep_ayer_list = []
        for layer in list(zip(deep_layers[:-1], deep_layers[1:])):
            # 全连接层，输入和输出维度分别是 layer[0] 和 layer[1]
            deep_ayer_list.append(nn.Linear(layer[0], layer[1]))
            # 批归一化层
            deep_ayer_list.append(nn.BatchNorm1d(layer[1], affine=False))
            # ReLU 激活函数
            deep_ayer_list.append(nn.ReLU(inplace=True))
        # 打包成一个 nn.Sequential 模块
        self._deep = nn.Sequential(*deep_ayer_list)

    def forward(self, x):
        out = self._deep(x)
        return out


class Cross(nn.Module):
    """
    x_0 * x_l^T * w_l + x_l + b_l ， x_0 是最初的输入
    """

    def __init__(self, input_dim, num_cross_layers):
        super(Cross, self).__init__()

        self.num_cross_layers = num_cross_layers
        weight_w = []
        weight_b = []
        batchnorm = []
        for i in range(num_cross_layers):
            # 用正态分布初始化权重。
            weight_w.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            weight_b.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            # 用于对一维特征数据进行归一化操作
            batchnorm.append(nn.BatchNorm1d(input_dim, affine=False))
        # nn.ParameterList用来存储和管理多层的可学习参数。
        self.weight_w = nn.ParameterList(weight_w)
        self.weight_b = nn.ParameterList(weight_b)
        self.bn = nn.ModuleList(batchnorm)

    def forward(self, x):
        out = x
        x = x.reshape(x.shape[0], -1, 1)
        for i in range(self.num_cross_layers):
            # torch.transpose(out, 1, 2)转置 out
            xTw = torch.matmul(
                torch.transpose(out.reshape(out.shape[0], -1, 1), 1, 2),
                self.weight_w[i].reshape(1, -1, 1),
            )
            xxTw = torch.matmul(x, xTw)
            xxTw = xxTw.reshape(xxTw.shape[0], -1)
            out = xxTw + self.weight_b[i] + out
            # 的作用是对当前层的输出 out 进行批归一化。
            out = self.bn[i](out)
        return out, xTw
