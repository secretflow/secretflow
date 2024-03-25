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

import torch
from torchvision import datasets, transforms

from .base_dataset import BaseDataset
from .split_dataset import ActiveDataset, PassiveDataset


class CIFARDataset(BaseDataset):
    def __init__(self, dataset_name, data_path, args={}):
        super().__init__(dataset_name, data_path)

        self.args = args

        # initialize the transformer
        self.composers = self.args.get("composers", None)
        self.train_transform = None
        self.valid_transform = None
        self.init_transform(self.composers)

        # initialize train and valid dataset
        self.load_dataset()

    def init_transform(self, composers):
        if composers is None:
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                          std=[0.229, 0.224, 0.225])

            self.train_transform = transforms.Compose(
                [
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    # normalize
                ]
            )

            self.valid_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # normalize
                ]
            )
        else:
            self.train_transform = composers[0]
            self.valid_transform = composers[1]

    def load_dataset(self):
        if self.dataset_name == "cifar10":
            self.train_dataset = datasets.CIFAR10(
                self.data_path,
                train=True,
                download=False,
                transform=self.train_transform,
            )

            self.valid_dataset = datasets.CIFAR10(
                self.data_path,
                train=False,
                download=False,
                transform=self.valid_transform,
            )
        else:
            self.train_dataset = datasets.CIFAR100(
                self.data_path,
                train=True,
                download=False,
                transform=self.train_transform,
            )

            self.valid_dataset = datasets.CIFAR100(
                self.data_path,
                train=False,
                download=False,
                transform=self.valid_transform,
            )

        if isinstance(self.train_dataset, torch.utils.data.dataset.Subset):
            label_dict = self.train_dataset.dataset.class_to_idx
        else:
            label_dict = self.train_dataset.class_to_idx

        self.label_set = [label for name, label in label_dict.items()]

    def _split_data(self, dataset, party_num=2, channel_first=True):
        parties = {}
        for party_index in range(party_num):
            parties[party_index] = []

        # split data
        labels, indexes = [], []
        interval = dataset[0][0].shape[-1] // party_num

        for index, (tensor, label) in enumerate(dataset):
            if not channel_first:
                tensor = tensor.permute(1, 2, 0)

            for i in range(party_num - 1):
                ntensor = tensor[:, i * interval : (i + 1) * interval, :]
                parties[i].append(ntensor.unsqueeze(0))

            ntensor = tensor[:, (party_num - 1) * interval :, :]
            parties[party_num - 1].append(ntensor.unsqueeze(0))
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
