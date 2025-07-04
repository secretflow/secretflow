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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchmetrics import Accuracy, Precision

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.applications.sl_resnet_torch import NaiveSumSoftmax
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.attacks.batch_level_lia_torch import (
    BatchLevelLabelInferenceAttack,
)

from .model_def import BottomModelForCifar10


class CIFARSIMDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.rand(size, 3, 32, 32)
        self.targets = [random.randint(0, 9) for i in range(size)]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.size


def do_test_sl_and_lia(alice, bob):
    device_y = bob
    bs = 10

    train_dataset = CIFARSIMDataset(20)
    train_data = train_dataset.data.numpy()
    train_label = np.array(train_dataset.targets)

    # put into FedNdarray
    fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, :, :, 0:16])(train_data),
            bob: bob(lambda x: x[:, :, :, 16:32])(train_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label = bob(lambda x: x)(train_label)

    # model configure
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=1e-3)
    base_model = TorchModel(
        model_fn=BottomModelForCifar10,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average="micro"
            ),
        ],
    )

    fuse_model = TorchModel(
        model_fn=NaiveSumSoftmax,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average="micro"
            ),
        ],
        use_softmax=False,
    )

    base_model_dict = {
        alice: base_model,
        bob: base_model,
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=fuse_model,
        dp_strategy_dict=None,
        compressor=None,
        simulation=True,
        random_seed=1234,
        backend="torch",
        strategy="split_nn",
    )

    eval_label = train_dataset.targets

    dummy_fuse_model = TorchModel(
        model_fn=NaiveSumSoftmax,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average="micro"
            ),
        ],
        use_softmax=False,
    )

    lia_cb = BatchLevelLabelInferenceAttack(
        attack_party=alice,
        victim_party=bob,
        victim_hidden_size=[10],
        dummy_fuse_model=dummy_fuse_model,
        exec_device="cpu",
        label=eval_label,
        epochs=10,
    )

    history = sl_model.fit(
        fed_data,
        label,
        epochs=1,
        batch_size=bs,
        shuffle=False,
        random_seed=1234,
        callbacks=[lia_cb],
    )
    print(history)

    metric = lia_cb.get_attack_metrics()
    print(metric)


def test_sl_and_lia(sf_simulation_setup_devices):
    devices = sf_simulation_setup_devices
    do_test_sl_and_lia(devices.alice, devices.bob)
