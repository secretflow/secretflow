#!/usr/bin/env python
# coding=utf-8
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

import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FuseModelNaive(nn.Module):
    def __init__(self, input_shapes, output_shape):
        super().__init__()

    def forward(self, x):
        return x


class FuseModelSum(nn.Module):
    def __init__(self, input_shapes, output_shape):
        super().__init__()
        self.linear = nn.Linear(output_shape, output_shape)

    def forward(self, x):
        return self.linear(F.relu(x))


class FuseModelCat(nn.Module):
    def __init__(self, input_shapes, output_shape):
        super().__init__()
        self.linear = nn.Linear(np.sum(input_shapes), output_shape)

    def forward(self, x):
        return self.linear(F.relu(x))


def get_fuse_model(input_shapes, output_shape, aggregation):
    if aggregation == "naive_sum":
        model = FuseModelNaive(input_shapes, output_shape)
    elif aggregation == "sum":
        model = FuseModelSum(input_shapes, output_shape)
    elif aggregation == "concatenate":
        model = FuseModelCat(input_shapes, output_shape)
    else:
        raise TypeError("Invalid aggregation method!!!")
    return model
