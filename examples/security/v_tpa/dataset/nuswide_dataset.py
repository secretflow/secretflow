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

import os
import pdb

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset

from .base_dataset import BaseDataset
from .split_dataset import ActiveDataset, PassiveDataset


class NUSWIDEDataset(BaseDataset):
    def __init__(self, dataset_name, data_path, args={}):
        super().__init__(dataset_name, data_path)

        self.args = args
        self.valid_ratio = self.args.get("valid_ratio", 0.2)

        self.train_dataset = None
        self.valid_dataset = None
        self.label_set = None
        self.load_dataset()

        self.split_points = {
            1: [0, 64 + 225 + 144 + 73 + 128 + 1000],
            2: [0, 64 + 225 + 144 + 73 + 128, 64 + 225 + 144 + 73 + 128 + 1000],
            3: [0, 64, 64 + 225 + 144 + 73 + 128, 64 + 225 + 144 + 73 + 128 + 1000],
            4: [
                0,
                64,
                64 + 225,
                64 + 225 + 144 + 73 + 128,
                64 + 225 + 144 + 73 + 128 + 1000,
            ],
            5: [
                0,
                64,
                64 + 225,
                64 + 225 + 144,
                64 + 225 + 144 + 73 + 128,
                64 + 225 + 144 + 73 + 128 + 1000,
            ],
            6: [
                0,
                64,
                64 + 225,
                64 + 225 + 144,
                64 + 225 + 144 + 73,
                64 + 225 + 144 + 73 + 128,
                64 + 225 + 144 + 73 + 128 + 1000,
            ],
        }

    def load_dataset(self):
        train_feature_file = os.path.join(self.data_path, "train_features.txt")
        train_features = np.loadtxt(train_feature_file, delimiter=",", dtype=np.float32)

        train_label_file = os.path.join(self.data_path, "train_label_idxes.txt")
        train_labels = np.loadtxt(train_label_file, dtype=np.int32)

        valid_feature_file = os.path.join(self.data_path, "valid_features.txt")
        valid_features = np.loadtxt(valid_feature_file, delimiter=",", dtype=np.float32)

        valid_label_file = os.path.join(self.data_path, "valid_label_idxes.txt")
        valid_labels = np.loadtxt(valid_label_file, dtype=np.int32)

        self.train_dataset = TensorDataset(
            torch.from_numpy(train_features).float(),
            torch.from_numpy(train_labels).int(),
        )
        self.valid_dataset = TensorDataset(
            torch.from_numpy(valid_features).float(),
            torch.from_numpy(valid_labels).int(),
        )
        self.label_set = np.unique(train_labels).tolist()

    def _split_data(self, dataset, party_num=2, channel_first=True):
        if party_num not in self.split_points:
            raise ValueError("Invalid number of participants!!!")

        parties = {}
        for party_index in range(party_num):
            parties[party_index] = []

        # split data
        labels, indexes = [], []
        points = self.split_points[party_num]
        for index, (tensor, label) in enumerate(dataset):
            for i in range(party_num):
                parties[i].append(tensor[points[i] : points[i + 1]].unsqueeze(0))

            indexes.append(torch.LongTensor([index]))
            labels.append(torch.LongTensor([label]))

        labels = torch.cat(labels)
        indexes = torch.cat(indexes)
        for party_index in range(party_num):
            parties[party_index] = torch.cat(parties[party_index])

        pdatasets = []
        for party_index in range(party_num):
            pdatasets.append(PassiveDataset(parties[party_index], labels, indexes))
        adataset = ActiveDataset(None, labels, indexes)
        return pdatasets, adataset
