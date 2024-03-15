#!/usr/bin/env python
# coding=utf-8
import torch
from torchvision import datasets, transforms

from .base_dataset import BaseDataset
from .split_dataset import ActiveDataset, PassiveDataset


class MNISTDataset(BaseDataset):
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
            transform = transforms.Compose([transforms.ToTensor()])
            self.train_transform = transform
            self.valid_transform = transform
        else:
            self.train_transform = composers[0]
            self.valid_transform = composers[1]

    def load_dataset(self):
        self.train_dataset = datasets.MNIST(
            self.data_path, train=True, download=False, transform=self.train_transform
        )

        self.valid_dataset = datasets.MNIST(
            self.data_path, train=False, download=False, transform=self.valid_transform
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
        is_3d = len(dataset[0][0].shape) == 3

        for index, (tensor, label) in enumerate(dataset):
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
