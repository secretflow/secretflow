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


class BadNetsBaseDataset:
    def __init__(self, train_dataset, valid_dataset, badnets_args):
        self.train_poison_ratio = badnets_args["train_poison_ratio"]
        self.valid_poison_ratio = badnets_args["valid_poison_ratio"]
        self.target_class = badnets_args["target_class"]
        self.train_known_target_num = badnets_args["train_known_target_num"]
        self.valid_known_target_num = badnets_args["valid_known_target_num"]
        if self.train_poison_ratio > 0.0:
            self.train_poisoning_indexes = np.random.choice(
                range(len(self.train_dataset)),
                int(self.train_poison_ratio * len(self.train_dataset)),
                replace=False,
            )
        else:
            self.train_poisoning_indexes = []

        if self.valid_poison_ratio > 0.0:
            self.valid_poisoning_indexes = np.random.choice(
                range(len(self.valid_dataset)),
                int(self.valid_poison_ratio * len(self.valid_dataset)),
                replace=False,
            )
        else:
            self.valid_poisoning_indexes = []
        self.train_target_indexes = self.build_target_set(
            train_dataset, self.train_known_target_num
        )
        self.valid_target_indexes = self.build_target_set(
            valid_dataset, self.valid_known_target_num
        )

    def build_target_set(self, dataset, known_target_num):
        if isinstance(dataset, torch.utils.data.dataset.TensorDataset):
            targets = dataset.tensors[-1]
        else:
            targets = dataset.targets
        target_indexes = np.where(np.array(targets) == self.target_class)[0]
        target_set = np.random.choice(target_indexes, known_target_num, replace=False)
        return target_set

    def get_train_poisoning_indexes(self):
        return self.train_poisoning_indexes

    def get_valid_poisoning_indexes(self):
        return self.valid_poisoning_indexes

    def get_train_target_indexes(self):
        return self.train_target_indexes

    def get_valid_target_indexes(self):
        return self.valid_target_indexes
