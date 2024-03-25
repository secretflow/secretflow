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

import random

from torch.utils.data import Sampler


class ShuffleSampler(Sampler):
    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.n = len(self.data_source)
        self.indices = list(range(self.n))
        self.seed = seed

    def __iter__(self):
        random.seed(self.seed)
        random.shuffle(self.indices)

        return iter(self.indices)

    def __len__(self):
        return self.n
