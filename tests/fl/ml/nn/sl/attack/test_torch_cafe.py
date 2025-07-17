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
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Precision

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.attacks.cafe_torch import CAFEAttack

from .model_def import CafeServer, LocalEmbedding


def do_test_sl_and_cafe(alice, bob, carol):
    device_y = carol
    bs = 40
    total_num = 80
    # fake data
    train_label = np.random.randint(10, size=total_num)
    train_data = np.random.randint(0, 256, size=(total_num, 28, 28)).astype(np.float32)
    train_data = train_data / 255
    data_for_save_tensor = torch.tensor(train_data)

    data_for_save = [
        data_for_save_tensor[:, 0:14, 0:14],
        data_for_save_tensor[:, 0:14, 14:28],
    ]

    fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, 0:14, 0:14])(train_data),
            bob: bob(lambda x: x[:, 0:14, 14:28])(train_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label = carol(lambda x: x)(train_label)

    # model configure
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=1e-3)
    base_model = TorchModel(
        model_fn=LocalEmbedding,
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

    fuse_model = TorchModel(
        model_fn=CafeServer,
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
        clients_num=2,
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
        backend='torch',
        strategy='split_nn',
    )

    cafe_attack = CAFEAttack(
        attack_party=carol,
        label_party=carol,
        victim_hidden_size=[10],
        exec_device='cpu',
        real_data_for_save=data_for_save,
        number_of_workers=2,
    )

    history = sl_model.fit(
        fed_data,
        label,
        epochs=2000,
        batch_size=bs,
        shuffle=False,
        random_seed=1234,
        callbacks=[cafe_attack],
    )
    print(history)


@pytest.mark.skip(reason="FIXME: broken OSCP code, tasks too long time")
def test_sl_and_cafe(sf_simulation_setup_devices):
    devices = sf_simulation_setup_devices
    do_test_sl_and_cafe(devices.alice, devices.bob, devices.carol)
