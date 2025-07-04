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

import logging

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from secretflow_fl.ml.nn.applications.sl_resnet_torch import (
    BasicBlock,
    ResNetBase,
    ResNetFuse,
)
from secretflow_fl.ml.nn.applications.sl_vgg_torch import VGGBase, VGGFuse
from secretflow_fl.ml.nn.core.torch import BaseModule


class SimSLVGG16(BaseModule):
    def __init__(self):
        super(SimSLVGG16, self).__init__()
        self.alice_base = VGGBase(use_passport=True)
        self.bob_base = VGGBase(use_passport=True)
        self.fuse = VGGFuse(use_passport=True)

    def forward(self, x):
        alice_hid = self.alice_base(x[0])
        bob_hid = self.bob_base(x[1])
        out = self.fuse((alice_hid, bob_hid))
        return out


class SimSLResNet18(BaseModule):
    def __init__(self):
        super(SimSLResNet18, self).__init__()
        self.alice_base = ResNetBase(BasicBlock, [2, 2, 2, 2], use_passport=True)
        self.bob_base = ResNetBase(BasicBlock, [2, 2, 2, 2], use_passport=True)
        self.fuse = ResNetFuse(use_passport=True)

    def forward(self, x):
        alice_hid = self.alice_base(x[0])
        bob_hid = self.bob_base(x[1])
        out = self.fuse((alice_hid, bob_hid))
        return out


class CustomDataset(Dataset):
    def __init__(self, data, labels, split_idx):
        self.data = data
        self.labels = labels
        self.split_idx = split_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_alice = self.data[index][..., : self.split_idx]
        data_bob = self.data[index][..., self.split_idx :]
        label = self.labels[index]
        return (data_alice, data_bob), label


def simulate_sl_model_training(model):
    data_num = 20
    data = torch.randn(data_num, 3, 32, 32)
    labels = torch.randint(0, 10, (data_num,))

    custom_dataset = CustomDataset(data, labels, split_idx=16)

    batch_size = 4
    data_loader = DataLoader(
        dataset=custom_dataset, batch_size=batch_size, shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for batch_id, (data, label) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        logging.info(f"batch {batch_id}, loss: {loss}")


def test_sl_cv_model_vgg():
    vgg = SimSLVGG16()
    simulate_sl_model_training(vgg)


def test_sl_cv_model_resnet():
    resnet = SimSLResNet18()
    simulate_sl_model_training(resnet)
