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
from torch import nn as nn
from torch import optim
from torchmetrics import AUROC, Accuracy, Precision

from secretflow.ml.nn.core.torch import BaseModule


class ConvNetBase(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetBase, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )

        # fully connected layer, output 10 classes
        self.out = nn.Linear(192, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def output_num(self):
        return 1


class ConvNetFuse(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetFuse, self).__init__()
        self.fc1 = nn.Linear(64 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ConvNetFuseAgglayer(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetFuseAgglayer, self).__init__()
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ConvNetRegBase(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetRegBase, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )

        # fully connected layer, output 10 classes
        self.out = nn.Linear(192, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.sum(torch.abs(param))

        return output, reg_loss

    def output_num(self):
        return 2

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)

    def configure_metrics(self):
        """no use for sl base model."""
        return None

    def configure_loss(self):
        """no use for sl base model."""
        return None


class ConvNetRegFuse(BaseModule):
    """Small ConvNet basenet for MNIST."""

    def __init__(self):
        super(ConvNetRegFuse, self).__init__()
        self.fc1 = nn.Linear(64 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, hiddens):
        x = hiddens[::2]
        x = torch.cat(x, dim=1)
        x = x.view(-1, 128)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx: int, dataloader_idx: int = 0, **kwargs):
        hiddens, y = batch
        out = self(hiddens)

        self.update_metrics(out, y)

        reg_loss = hiddens[1::2]
        reg_loss = reg_loss[0] + reg_loss[1]
        for param in self.parameters():
            reg_loss += torch.sum(torch.abs(param))

        loss = self.loss(out, y) + 1e-4 * reg_loss
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)

    def configure_metrics(self):
        return [
            Accuracy(task="multiclass", num_classes=10, average='micro'),
            Precision(task="multiclass", num_classes=10, average='micro'),
            AUROC(task="multiclass", num_classes=10),
        ]

    def configure_loss(self):
        return nn.CrossEntropyLoss()
