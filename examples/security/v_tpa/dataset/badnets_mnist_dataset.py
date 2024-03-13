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

import sys

sys.path.append("..")

import pdb

import torch
from attack.badnets.trigger import inject_mnist_trigger, inject_white_trigger

from .mirror_mnist_dataset import MirrorMNISTDataset
from .split_dataset import ActiveDataset, PassiveDataset


class BadNetsMNISTDataset(MirrorMNISTDataset):
    def __init__(self, dataset_name, data_path, args={}, badnets_args={}):
        super().__init__(dataset_name, data_path, args, badnets_args)
        self.trigger_size = badnets_args.get("trigger_size", 4)

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

    def _split_data(self, dataset, poisoning_indexes, party_num=2, channel_first=True):
        parties = {}
        for party_index in range(party_num):
            parties[party_index] = []

        # split data
        labels, indexes = [], []
        interval = dataset[0][0].shape[-1] // party_num
        is_3d = len(dataset[0][0].shape) == 3

        for index, (tensor, label) in enumerate(dataset):
            # inject trigger
            if index in poisoning_indexes:
                # tensor = inject_mnist_trigger(tensor)
                tensor = inject_white_trigger(tensor, self.trigger_size)

            if not channel_first and is_3d:
                tensor = tensor.permute(1, 2, 0)

            for i in range(party_num - 1):
                if is_3d:
                    parties[i].append(
                        tensor[:, i * interval : (i + 1) * interval, :]
                        .flatten()
                        .unsqueeze(0)
                    )
                else:
                    parties[i].append(
                        tensor[i * interval : (i + 1) * interval, :]
                        .flatten()
                        .unsqueeze(0)
                    )

            if is_3d:
                parties[party_num - 1].append(
                    tensor[:, (party_num - 1) * interval :, :].flatten().unsqueeze(0)
                )
            else:
                parties[party_num - 1].append(
                    tensor[(party_num - 1) * interval :, :].flatten().unsqueeze(0)
                )

            indexes.append(torch.LongTensor([index]))
            labels.append(torch.LongTensor([label]))

        # concatenate different portions
        labels = torch.cat(labels)
        indexes = torch.cat(indexes)
        for party_index in range(party_num):
            parties[party_index] = torch.cat(parties[party_index])

        # create the passive and activate datasets
        pdatasets = []
        for party_index in range(party_num):
            pdatasets.append(PassiveDataset(parties[party_index], labels, indexes))
        adataset = ActiveDataset(None, labels, indexes)
        return pdatasets, adataset
