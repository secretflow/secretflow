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

import numpy as np
from torch import nn


class SGDRScheduler(nn.Module):
    global_epoch = 0
    all_epoch = 0
    cur_drop_prob = 0.0

    def __init__(self, dropblock):
        super(SGDRScheduler, self).__init__()
        self.dropblock = dropblock
        self.drop_values = 0.0

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        ix = np.log2(self.global_epoch / 10 + 1).astype(np.int)
        T_cur = self.global_epoch - 10 * (2 ** (ix) - 1)
        T_i = 10 * 2**ix
        self.dropblock.drop_prob = np.abs(
            (0 + 0.5 * 0.1 * (1 + np.cos(np.pi * T_cur / T_i))) - 0.1
        )
        SGDRScheduler.cur_drop_prob = self.dropblock.drop_prob


class LinearScheduler(nn.Module):
    global_epoch = 0
    num_epochs = 0

    def __init__(self, dropblock, start_value=0.0, stop_value=0.1):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.drop_values = np.linspace(
            start=start_value, stop=stop_value, num=self.num_epochs
        )

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        self.dropblock.drop_prob = self.drop_values[self.global_epoch]
