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

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class CIFARSIMLabeled(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.rand(size, 3, 32, 16)
        self.targets = [random.randint(0, 9) for i in range(size)]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.size


class CIFARSIMUnlabeled(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.rand(size, 3, 32, 16)
        self.targets = [-1] * size

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.size


class CIFAR10Labeled(datasets.CIFAR10):
    def __init__(
        self,
        root,
        indexs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(CIFAR10Labeled, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        img = img[:, :, :16]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10Unlabeled(CIFAR10Labeled):
    def __init__(
        self,
        root,
        indexs,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(CIFAR10Unlabeled, self).__init__(
            root,
            indexs,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.targets = np.array([-1 for i in range(len(self.targets))])
        self.data = self.data[:, :, :, :16]


def label_index_split(labels, n_labeled_per_class, num_classes):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    return train_labeled_idxs, train_unlabeled_idxs
