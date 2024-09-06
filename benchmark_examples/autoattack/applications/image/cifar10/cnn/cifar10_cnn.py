# Copyright 2024 Ant Group Co., Ltd.
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

from typing import Any

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack.applications.base import ModelType
from benchmark_examples.autoattack.applications.image.cifar10.cifar10_base import (
    Cifar10ApplicationBase,
)
from benchmark_examples.autoattack.utils.resources import ResourceDict, ResourcesPack
from secretflow.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)


class CnnBase(BaseModule):
    def __init__(self, preprocess_layer=None):
        super().__init__()
        self.class_num = 10
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, padding=1, stride=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.norm = nn.BatchNorm2d(18)
        self.preprocess_layer = preprocess_layer

    def forward(self, x):
        if self.preprocess_layer is not None:
            x = self.preprocess_layer(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.norm(x)
        return torch.flatten(x, 1)

    def output_num(self):
        return 1


class CnnFuse(BaseModule):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x) -> Any:
        x = torch.cat(x, dim=1)
        x = x.view(-1, 18 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Cifar10CNN(Cifar10ApplicationBase):
    def __init__(self, alice, bob):
        super().__init__(
            alice,
            bob,
            train_batch_size=128,
            hidden_size=2304,
            dnn_fuse_units_size=[512 * 2],
            epoch=10,
        )
        self.metrics = [
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(AUROC, task="multiclass", num_classes=10),
        ]

    def model_type(self) -> ModelType:
        return ModelType.CNN

    def _create_base_model(self):
        return TorchModel(
            model_fn=CnnBase,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
        )

    def dnn_fuse_units_size_range(self):
        return [
            [512 * 2],
            [512 * 2, 512],
        ]

    def create_base_model_alice(self):
        return self._create_base_model()

    def create_base_model_bob(self):
        return self._create_base_model()

    def create_fuse_model(self):
        return TorchModel(
            model_fn=CnnFuse,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-3),
            metrics=self.metrics,
        )

    def resources_consumption(self) -> ResourcesPack:
        # 640MiB
        return (
            ResourcesPack()
            .with_debug_resources(
                ResourceDict(
                    gpu_mem=800 * 1024 * 1024, CPU=1, memory=3 * 1024 * 1024 * 1024
                )
            )
            .with_sim_resources(
                self.device_y.party,
                ResourceDict(
                    gpu_mem=800 * 1024 * 1024, CPU=1, memory=3 * 1024 * 1024 * 1024
                ),
            )
            .with_sim_resources(
                self.device_f.party,
                ResourceDict(
                    gpu_mem=600 * 1024 * 1024, CPU=1, memory=3 * 1024 * 1024 * 1024
                ),
            )
        )
