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

import torch

from .mirror_nuswide_dataset import MirrorNUSWIDEDataset
from .split_dataset import ActiveDataset, PassiveDataset


class BadNetsNUSWIDEDataset(MirrorNUSWIDEDataset):
    def __init__(self, dataset_name, data_path, args={}, badnets_args={}):
        super().__init__(dataset_name, data_path, args, badnets_args)

    def split_train(self, party_num=2, channel_first=True):
        if self.train_pdatasets is None:
            self.train_pdatasets, self.train_adataset = self._split_data(
                self.train_dataset,
                self.train_poisoning_indexes,
                party_num,
                channel_first,
            )
        return self.train_pdatasets, self.train_adataset

    def split_valid(self, party_num=2, channel_first=True):
        if self.valid_pdatasets is None:
            self.valid_pdatasets, self.valid_adataset = self._split_data(
                self.valid_dataset,
                self.valid_poisoning_indexes,
                party_num,
                channel_first,
            )
        return self.valid_pdatasets, self.valid_adataset

    def inject_trigger(self, tensor):
        tensor[-1] = 1.0
        return tensor

    def _split_data(self, dataset, poisoning_indexes, party_num=2, channel_first=True):
        if party_num not in self.split_points:
            raise ValueError("Invalid number of participants!!!")

        parties = {}
        for party_index in range(party_num):
            parties[party_index] = []

        # split data
        labels, indexes = [], []
        points = self.split_points[party_num]
        for index, (tensor, label) in enumerate(dataset):
            if index in poisoning_indexes:
                tensor = self.inject_trigger(tensor)

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
