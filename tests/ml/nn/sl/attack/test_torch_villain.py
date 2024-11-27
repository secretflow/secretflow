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
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, Precision

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.device import reveal
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.applications.sl_resnet_torch import (
    BasicBlock,
    NaiveSumSoftmax,
    ResNetBase,
)
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.attacks.villain_attack_torch import VillainAttack


class ImageDatasetLeft(Dataset):
    def __init__(self, data_num):
        self.data = torch.randn(data_num, 3, 32, 16)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data


class ImageDatasetRight(Dataset):
    def __init__(self, data_num, return_y):
        self.data = torch.randn(data_num, 3, 32, 16)
        self.labels = torch.randint(0, 10, (data_num,))
        self.return_y = return_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        if self.return_y:
            return (data), label
        return data


def create_dataset_builder(
    data_num,
    batch_size=32,
    is_left=True,
    return_y=True,
):
    def dataset_builder(x):
        if is_left:
            dataset = ImageDatasetLeft(data_num)
        else:
            dataset = ImageDatasetRight(data_num, return_y)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return dataloader

    return dataset_builder


def create_classifier():
    layers = [nn.Linear(512, 10)]
    classifier = nn.Sequential(*layers)
    return classifier


@pytest.mark.skip(reason='FIXME: broken OSCP code, throw error')
def test_villain_attack(sf_simulation_setup_devices):
    target_class = 6
    data_num = 40
    target_set = [1, 9, 17, 25]  # 攻击目标样本索引
    batch_size = 8

    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    device_y = bob

    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(optim.Adam, lr=1e-3)

    base_model = TorchModel(
        model_fn=ResNetBase,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(AUROC, task="multiclass", num_classes=10),
        ],
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        classifier=create_classifier(),
    )

    fuse_model = TorchModel(
        model_fn=NaiveSumSoftmax,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average='micro'
            ),
            metric_wrapper(AUROC, task="multiclass", num_classes=10),
        ],
    )

    base_model_dict = {alice: base_model, bob: base_model}
    dataset_builder_dict = {
        alice: create_dataset_builder(
            data_num=data_num, batch_size=batch_size, is_left=True
        ),
        bob: create_dataset_builder(
            data_num=data_num, batch_size=batch_size, is_left=False
        ),
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=fuse_model,
        simulation=True,
        random_seed=1234,
        backend='torch',
        strategy='split_nn',
    )

    train_data = np.random.rand(data_num, 3, 32, 32)
    train_label = np.random.randint(0, 10, size=(data_num,))
    data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, :, :, 0:16])(train_data),
            bob: bob(lambda x: x[:, :, :, 16:32])(train_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label = bob(lambda x: x)(train_label)

    villain_attack = VillainAttack(
        attack_party=alice,
        origin_target_idx=target_set[0],
        theta=0.5,
    )

    history = sl_model.fit(
        data,
        label,
        epochs=10,
        batch_size=batch_size,
        shuffle=False,
        random_seed=1234,
        dataset_builder=dataset_builder_dict,
        callbacks=[villain_attack],
    )
    print(history)

    global_metric = sl_model.evaluate(
        data,
        label,
        batch_size=128,
        dataset_builder=dataset_builder_dict,
        callbacks=[villain_attack],
    )
    print(global_metric)

    pred_dataset_builder_dict = {
        alice: create_dataset_builder(
            data_num=data_num, batch_size=batch_size, is_left=True
        ),
        bob: create_dataset_builder(
            data_num=data_num, batch_size=batch_size, is_left=False, return_y=False
        ),
    }
    preds = sl_model.predict(
        data,
        batch_size=128,
        dataset_builder=pred_dataset_builder_dict,
        callbacks=[villain_attack],
    )

    preds_plain = []
    for pred in preds:
        preds_plain.append(torch.argmax(pred, dim=1))
    preds_plain = torch.cat(preds_plain, dim=0)

    pred_np: np.ndarray = preds_plain.cpu().numpy()
    poison_pred = pred_np
    poison_pred = poison_pred == target_class
    nums_poison_true = sum(poison_pred)
    acc = nums_poison_true / len(poison_pred)
    print("acc: " + acc)
