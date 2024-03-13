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

from .badnets_base_dataset import BadNetsBaseDataset
from .mnist_dataset import MNISTDataset


class MirrorMNISTDataset(MNISTDataset, BadNetsBaseDataset):
    def __init__(self, dataset_name, data_path, args={}, mirror_args={}):
        MNISTDataset.__init__(self, dataset_name, data_path, args)
        BadNetsBaseDataset.__init__(
            self, self.train_dataset, self.valid_dataset, mirror_args
        )
