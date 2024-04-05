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

from torch.utils.data import Dataset


class PassiveDataset(Dataset):
    def __init__(self, party_features, party_labels, party_indexes):
        super().__init__()

        self.party_features = party_features
        self.party_labels = party_labels
        self.party_indexes = party_indexes

    def __getitem__(self, index):
        return self.party_indexes[index], self.party_features[index]

    def __len__(self):
        return len(self.party_features)


class ActiveDataset(PassiveDataset):
    def __init__(self, party_features, party_labels, party_indexes):
        super().__init__(party_features, party_labels, party_indexes)

    def __getitem__(self, index):
        return self.party_indexes[index], self.party_labels[index]

    def __len__(self):
        return len(self.party_labels)
