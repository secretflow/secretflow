#!/usr/bin/env python
# coding=utf-8
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
        raise "Invalid aggregation method!!!"
    return model
