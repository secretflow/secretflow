#!/usr/bin/env python
# coding=utf-8
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
