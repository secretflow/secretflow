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

sys.path.append(".")

from .badnets_cifar_dataset import BadNetsCIFARDataset
from .badnets_mnist_dataset import BadNetsMNISTDataset
from .badnets_nuswide_dataset import BadNetsNUSWIDEDataset
from .cifar_dataset import CIFARDataset
from .mirror_cifar_dataset import MirrorCIFARDataset
from .mirror_mnist_dataset import MirrorMNISTDataset
from .mirror_nuswide_dataset import MirrorNUSWIDEDataset
from .mnist_dataset import MNISTDataset
from .nuswide_dataset import NUSWIDEDataset

DATASETS = {
    "cifar10": {
        "data_path": "<local dataset path>",  # change to the local dataset directory
        "normal": CIFARDataset,
        "mirror": MirrorCIFARDataset,
        "badnets": BadNetsCIFARDataset,
        "args": {"target_class": None},
    },
    "mnist": {
        "data_path": "<local dataset path>",  # change to the local dataset directory
        "normal": MNISTDataset,
        "mirror": MirrorMNISTDataset,
        "badnets": BadNetsMNISTDataset,
        "args": {"target_class": None},
    },
    "nus-wide": {
        "data_path": "<local dataset path>",  # change to the local dataset directory
        "normal": NUSWIDEDataset,
        "mirror": MirrorNUSWIDEDataset,
        "badnets": BadNetsNUSWIDEDataset,
        "args": {"target_class": 1},
    },
}
