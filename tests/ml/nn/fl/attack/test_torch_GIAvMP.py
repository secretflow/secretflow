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
from examples.security.h_gia.GIAvMP_attack.GIAvMP_torch import GIAvMP
from secretflow.security.aggregation import SecureAggregator
from secretflow_fl.ml.nn import FLModel
from secretflow_fl.ml.nn.core.torch import (
    BaseModule,
    TorchModel,
    metric_wrapper,
    optim_wrapper,
)
from secretflow_fl.utils.simulation.datasets_fl import (
    load_cifar10_horiontal,
    load_cifar10_unpartitioned,
)


# the FCNN model
class FCNNmodel(BaseModule):
    def __init__(self, classes=10):
        super(FCNNmodel, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc4(x)


# the CNN model
class CNNmodel(BaseModule):
    def __init__(self, classes=10):
        super(CNNmodel, self).__init__()
        self.act = nn.ReLU()
        self.body = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, classes)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


def do_test_fl_and_GIAvMP(attack_configs: dict, alice, bob):

    # prepare dataset
    client_data_num = attack_configs['k']
    (train_data, train_label), (test_data, test_label) = load_cifar10_horiontal(
        parts={alice: (0, client_data_num)},
        normalized_x=True,
        categorical_y=True,
        # is_torch=True,
    )

    # prepare aux dataset for attacker to train malicious params
    # here we use the testset of cifar10 as aux dataset, it can be replaced by other public dataset, such as imagenet
    ((x_train, y_train), (x_test, y_test)) = load_cifar10_unpartitioned()
    x_test = np.array(x_test, dtype=np.float32)
    aux_dataset = [i for i in zip(x_test, y_test)]

    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.SGD, lr=attack_configs['train_lr'])
    model_def = TorchModel(
        model_fn=attack_configs['model'],
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

    # init GIAvMP callback
    GIAvMP_callback = GIAvMP(
        attack_party=bob,
        victim_party=alice,
        aux_dataset=aux_dataset,
        attack_configs=attack_configs,
    )

    history = fl_model.fit(
        train_data,
        train_label,
        epochs=attack_configs['epochs'],
        batch_size=attack_configs['batchsize'],
        aggregate_freq=1,
        callbacks=[GIAvMP_callback],
    )

    return


# attack configurations
attack_configs = {
    "path_to_res": "./examples/security/h_gia/GIAvMP_attack/res",
    "path_to_trainedMP": "./examples/security/h_gia/GIAvMP_attack/malicious_params",
    "trainMP": True,
    # "trainMP": False,
    "dataset": "cifar10",
    "data_size": (3, 32, 32),
    "model": FCNNmodel,
    # "model": CNNmodel,
    "k": 8,
    "batchsize": 8,
    "epochs": 1,
    "train_lr": 1,
    "device": "cpu",
}


def test_fl_and_GIAvMP(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    do_test_fl_and_GIAvMP(attack_configs, alice, bob)


# sf.init(['alice', 'bob'], address='local', debug_mode=True)
# alice, bob = sf.PYU('alice'), sf.PYU('bob')
# do_test_fl_and_GIAvMP(attack_configs, alice, bob)
