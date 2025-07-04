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

import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch import optim
from torchmetrics import AUROC, Accuracy, Precision
from torchvision import transforms
from torchvision.datasets import CIFAR10

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.utils.simulation.datasets import _CACHE_DIR
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.attacks.sdar_torch import SDARAttack
from tests.fl.ml.nn.sl.attack.model_def import (
    Decoder,
    DecoderDiscriminator,
    FModel,
    GModel,
    SimulatorDiscriminator,
)

INTERMIDIATE_SHAPE = lambda level: (
    (16, 32, 32) if level == 3 else (32, 16, 16) if level < 7 else (64, 8, 8)
)
LEVEL = 4  # depth of split learning


def get_model():

    e_optim_fn = optim_wrapper(optim.Adam, lr=0.001, eps=1e-07)
    decoder_optim_fn = optim_wrapper(optim.Adam, lr=0.0005, eps=1e-07)
    simulator_d_optim_fn = optim_wrapper(optim.Adam, lr=4e-05, eps=1e-07)
    decoder_d_optim_fn = optim_wrapper(optim.Adam, lr=5e-09, eps=1e-07)
    g_optim_fn = optim_wrapper(optim.Adam, lr=0.001, eps=1e-07)
    f_optim_fn = optim_wrapper(optim.Adam, lr=0.001, eps=1e-07)

    loss_fn = nn.CrossEntropyLoss
    num_classes = 10
    return (
        TorchModel(
            model_fn=FModel,
            optim_fn=f_optim_fn,
            level=LEVEL,
            input_shape=(3, 32, 32),
        ),
        TorchModel(
            model_fn=GModel,
            loss_fn=loss_fn,
            optim_fn=g_optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
            level=LEVEL,
            input_shape=INTERMIDIATE_SHAPE(LEVEL),
            num_classes=10,
        ),
        TorchModel(
            model_fn=FModel,
            loss_fn=None,
            optim_fn=e_optim_fn,
            metrics=None,
            level=LEVEL,
            input_shape=(3, 32, 32),
        ),
        TorchModel(
            model_fn=Decoder,
            loss_fn=None,
            optim_fn=decoder_optim_fn,
            metrics=None,
            level=LEVEL,
            input_shape=INTERMIDIATE_SHAPE(LEVEL),
            num_classes=num_classes,
        ),
        TorchModel(
            model_fn=SimulatorDiscriminator,
            loss_fn=None,
            optim_fn=simulator_d_optim_fn,
            metrics=None,
            level=LEVEL,
            input_shape=INTERMIDIATE_SHAPE(LEVEL),
            num_classes=num_classes,
        ),
        TorchModel(
            model_fn=DecoderDiscriminator,
            loss_fn=None,
            optim_fn=decoder_d_optim_fn,
            metrics=None,
            input_shape=(3, 32, 32),
            num_classes=num_classes,
        ),
    )


class RepeatedDataset(torch_data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.len = len(x)
        self.transform = transform

    def __len__(self):
        return int(2**23)

    def __getitem__(self, idx):
        img = self.x[idx % self.len]
        label = self.y[idx % self.len]
        if self.transform:
            img = self.transform(img)
        return img, label


class OriginalDataset(torch_data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_data_builder(dataset_name='cifar10'):
    def data_builder():
        loader = CIFAR10
        data_dir = os.path.join(_CACHE_DIR, dataset_name)
        train_dataset = loader(
            data_dir, True, transform=transforms.ToTensor(), download=True
        )
        train_loader = torch_data.DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=False
        )
        train_data, train_labels = next(iter(train_loader))
        len_train_ds = len(train_data) // 2
        # evaluate with client data
        evaluate_data = train_data.numpy()[:len_train_ds]
        evaluate_label = train_labels.numpy()[:len_train_ds]

        train_plain_data = torch.tensor(train_data.numpy()[len_train_ds:])
        train_plain_label = torch.tensor(train_labels.numpy()[len_train_ds:])
        train_dataset = RepeatedDataset(train_plain_data, train_plain_label)
        train_loader = torch_data.DataLoader(
            train_dataset, batch_size=128, shuffle=False
        )
        evaluate_dataset = OriginalDataset(evaluate_data, evaluate_label)
        evaluate_loader = torch_data.DataLoader(
            evaluate_dataset, batch_size=128, shuffle=False
        )
        return train_loader, evaluate_loader

    return data_builder


def fill_parameters(alice, bob):
    def get_train_data():
        loader = CIFAR10
        data_dir = os.path.join(_CACHE_DIR, "cifar10")
        train_dataset = loader(
            data_dir, True, transform=transforms.ToTensor(), download=True
        )
        train_loader = torch_data.DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=False
        )
        train_data, train_labels = next(iter(train_loader))
        len_train_ds = len(train_data) // 2
        train_plain_data = train_data.numpy()[:len_train_ds]
        train_data = FedNdarray(
            partitions={
                alice: alice(lambda x: x)(train_plain_data),
                bob: bob(lambda x: x)(train_plain_data),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        sample_nums = 4
        train_data = train_data[0:sample_nums]
        return train_data

    def get_train_label():
        loader = CIFAR10
        data_dir = os.path.join(_CACHE_DIR, "cifar10")
        train_dataset = loader(
            data_dir, True, transform=transforms.ToTensor(), download=True
        )
        train_loader = torch_data.DataLoader(
            dataset=train_dataset, batch_size=len(train_dataset), shuffle=False
        )
        train_data, train_labels = next(iter(train_loader))
        len_train_ds = len(train_labels) // 2
        train_plain_label = train_labels.numpy()[:len_train_ds]
        train_label = bob(lambda x: x)(train_plain_label)
        sample_nums = 4
        train_label = train_label.device(lambda df: df[0:sample_nums])(train_label)
        return train_label

    def get_test_data():
        loader = CIFAR10
        data_dir = os.path.join(_CACHE_DIR, "cifar10")
        test_dataset = loader(
            data_dir, False, transform=transforms.ToTensor(), download=True
        )
        test_loader = torch_data.DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
        )
        test_data, test_labels = next(iter(test_loader))
        len_test_ds = len(test_data)
        test_plain_data = test_data.numpy()
        test_data = FedNdarray(
            partitions={
                alice: alice(lambda x: x)(test_plain_data),
                bob: bob(lambda x: x)(test_plain_data),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        sample_nums = 4
        test_data = test_data[0:sample_nums]
        return test_data

    def get_test_label():
        loader = CIFAR10
        data_dir = os.path.join(_CACHE_DIR, "cifar10")
        test_dataset = loader(
            data_dir, False, transform=transforms.ToTensor(), download=True
        )
        test_loader = torch_data.DataLoader(
            dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
        )
        test_data, test_labels = next(iter(test_loader))
        len_test_ds = len(test_labels)
        test_plain_label = test_labels.numpy()
        test_label = bob(lambda x: x)(test_plain_label)
        sample_nums = 4
        test_label = test_label.device(lambda df: df[0:sample_nums])(test_label)
        return test_label

    return get_train_data(), get_train_label(), get_test_data(), get_test_label()


def do_test_sl_and_sdar(alice, bob):
    device_y = bob
    f_model, g_model, e_model, decoder, simulator_d, decoder_d = get_model()
    fed_data, label, test_fed_data, test_data_label = fill_parameters(alice, bob)
    sdar_callback = SDARAttack(
        attack_party=bob,
        victim_party=alice,
        base_model_list=[alice],
        e_model_wrapper=e_model,
        decoder_model_wrapper=decoder,
        simulator_d_model_wrapper=simulator_d,
        decoder_d_model_wrapper=decoder_d,
        reconstruct_loss_builder=torch.nn.MSELoss,
        data_builder=get_data_builder(),
        exec_device='cpu',
    )
    sl_model = SLModel(
        base_model_dict={alice: f_model},
        device_y=device_y,
        model_fuse=g_model,
        dp_strategy_dict=None,
        compressor=None,
        simulation=True,
        random_seed=1234,
        backend="torch",
        strategy="split_nn",
    )
    sl_model.fit(
        fed_data,
        label,
        validation_data=(test_fed_data, test_data_label),
        epochs=1,
        batch_size=32,
        shuffle=False,
        random_seed=1234,
        dataset_builder=None,
        callbacks=[sdar_callback],
    )
    metrics = sdar_callback.get_attack_metrics()
    return metrics


def test_sl_and_sdar(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    do_test_sl_and_sdar(
        alice=alice,
        bob=bob,
    )
