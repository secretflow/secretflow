#!/usr/bin/env python
# coding=utf-8
import sys

sys.path.append("..")

import torch
import pdb
from attack.badnets.trigger import inject_cifar_trigger
from .mirror_cifar_dataset import MirrorCIFARDataset
from .split_dataset import PassiveDataset, ActiveDataset


class BadNetsCIFARDataset(MirrorCIFARDataset):
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

    def _split_data(self, dataset, poisoning_indexes, party_num=2, channel_first=True):
        parties = {}
        for party_index in range(party_num):
            parties[party_index] = []

        # split data
        labels, indexes = [], []
        interval = dataset[0][0].shape[-1] // party_num

        for index, (tensor, label) in enumerate(dataset):
            # inject trigger
            if index in poisoning_indexes:
                tensor = inject_cifar_trigger(tensor)

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
