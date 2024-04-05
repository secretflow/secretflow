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

import json
import pdb

import numpy as np


def asr(preds, truthes, target_class, indexes, hope_indexes):
    # t_indexes = truthes != target_class
    # p_indexes = preds == target_class
    # return np.sum(t_indexes & p_indexes) / np.sum(t_indexes)
    target_indexes = indexes[preds == target_class]
    correct = len(np.intersect1d(target_indexes, hope_indexes))
    return correct / len(hope_indexes)


def load_result(file):
    results = []
    with open(file, "r") as fp:
        for line in fp:
            r = json.loads(line)
            results.append(r)
    return results
