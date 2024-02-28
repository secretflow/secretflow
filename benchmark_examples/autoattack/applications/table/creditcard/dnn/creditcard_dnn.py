# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy

import secretflow as sf
from benchmark_examples.autoattack.applications.base import ApplicationBase
from secretflow.data.split import train_test_split
from secretflow.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper
from secretflow.preprocessing import StandardScaler
from secretflow.utils.simulation.datasets import load_creditcard


class CreditcardDnn(ApplicationBase):
    def __init__(self, config, alice, bob):
        super().__init__(
            config,
            alice,
            bob,
            device_y=bob,
            total_fea_nums=29,
            alice_fea_nums=25,
            num_classes=2,
            epoch=2,
            train_batch_size=1024,
            hidden_size=28,
            dnn_base_units_size_alice=[int(28 / 2), 28],
            dnn_base_units_size_bob=[4],
            dnn_fuse_units_size=[1],
        )

    def prepare_data(self):
        data = load_creditcard({self.alice: (0, 25), self.bob: (25, 29)})
        label = load_creditcard({self.bob: (29, 30)}).astype(np.float32)
        scaler = StandardScaler()
        data = scaler.fit_transform(data).astype(np.float32)
        random_state = 1234
        self.train_data, self.test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        self.train_label, self.test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )

    def create_base_model_alice(self):
        return TorchModel(
            model_fn=DnnBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
            ],
            input_dims=[25],
            dnn_units_size=self.dnn_base_units_size_alice,
        )

    def create_base_model_bob(self):
        return TorchModel(
            model_fn=DnnBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
            ],
            input_dims=[4],
            dnn_units_size=self.dnn_base_units_size_bob,
        )

    def create_fuse_model(self):
        return TorchModel(
            model_fn=DnnFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
            ],
            input_dims=[self.hidden_size, 4],
            dnn_units_size=[1],
        )

    def support_attacks(self):
        return ['norm', 'exploit']

    def alice_feature_nums_range(self) -> list:
        return [25]

    def hidden_size_range(self) -> list:
        return [28, 64]

    def dnn_base_units_size_range_alice(self) -> Optional[list]:
        return [
            [-0.5, -1],
            [-1],
            [
                -0.5,
                -1,
                -1,
            ],
        ]

    def dnn_base_units_size_range_bob(self) -> Optional[List[List[int]]]:
        return [[4]]

    def dnn_fuse_units_size_range(self) -> Optional[list]:
        return [
            [1],
            [-1, 1],
            [-1, -1, 1],
        ]

    def exploit_label_counts(self) -> Tuple[int, int]:
        label = sf.reveal(self.train_label.partitions[self.bob].data)
        neg, pos = np.bincount(label['Class'])
        return neg, pos
