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
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchmetrics import Accuracy, Precision

import secretflow as sf
from examples.security.h_gia.SME_attack.gia_sme_torch import GiadentInversionAttackSME
from secretflow.security.aggregation import SecureAggregator
from secretflow_fl.ml.nn import FLModel
from secretflow_fl.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from secretflow_fl.utils.simulation.datasets_fl import load_mnist


# the global model
class FLBaseNet(BaseModule):
    def __init__(self, classes=10):
        super(FLBaseNet, self).__init__()
        self.act = nn.ReLU()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(3136, 100)
        self.fc2 = nn.Linear(100, classes)

    def forward(self, x):
        x = self.body(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


def do_test_fl_and_gia_sme(attack_configs: dict, alice, bob):

    # prepare dataset
    client_data_num = attack_configs['k']
    (train_data, train_label), (test_data, test_label) = load_mnist(
        parts={alice: (0, client_data_num)},
        normalized_x=True,
        categorical_y=True,
        is_torch=True,
    )

    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=attack_configs['train_lr'])
    model_def = TorchModel(
        model_fn=FLBaseNet,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average='micro'
            ),
        ],
    )

    device_list = [alice]
    server = bob
    aggregator = SecureAggregator(server, [alice])

    # spcify params
    fl_model = FLModel(
        server=server,
        device_list=device_list,
        model=model_def,
        aggregator=aggregator,
        strategy='fed_avg_w',  # fl strategy
        backend="torch",  # backend support ['tensorflow', 'torch']
        # use_gpu=True,
    )

    # conduct SME attack through callback
    gia_callback = GiadentInversionAttackSME(
        attack_party=bob, victim_party=alice, attack_configs=attack_configs
    )

    history = fl_model.fit(
        train_data,
        train_label,
        epochs=attack_configs['epochs'],
        batch_size=attack_configs['batchsize'],
        aggregate_freq=1,
        callbacks=[gia_callback],
    )

    return


# attack configurations
attack_configs = {
    "path_to_res": "./examples/security/h_gia/SME_attack/res",
    "dataset": "MNIST",
    "k": 12,
    "batchsize": 6,
    "epochs": 10,
    "alpha": 0.5,
    "lamb": 0.1,
    "train_lr": 0.004,
    "eta": 0.5,
    "beta": 0.001,
    "iters": 1000,
    "test_steps": 100,
    "lr_decay": True,
    "save_figure": True,
    "setup": {"device": "cpu", "dtype": torch.float32},
    "mean_std": ((0.0,), (1.0,)),
}


def test_fl_and_gia_sme(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    do_test_fl_and_gia_sme(attack_configs, alice, bob)
