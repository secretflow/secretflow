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
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchmetrics import Accuracy, Precision
from torchvision import datasets, transforms

import secretflow as sf
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.attacks.lia_torch import LabelInferenceAttack

from .data_util import (
    CIFAR10Labeled,
    CIFAR10Unlabeled,
    CIFARSIMLabeled,
    CIFARSIMUnlabeled,
    label_index_split,
)
from .model_def import BottomModelForCifar10, BottomModelPlus, TopModelForCifar10


class CIFARSIMDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.rand(size, 3, 32, 32)
        self.targets = [random.randint(0, 9) for i in range(size)]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.size


def data_builder(batch_size, file_path=None):
    def prepare_data():
        n_labeled = 40
        num_classes = 10

        def get_transforms():
            transform_ = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            return transform_

        transforms_ = get_transforms()

        if file_path is None:
            # fake dataset, just for unittest
            train_labeled_dataset = CIFARSIMLabeled(200)
            train_unlabeled_dataset = CIFARSIMUnlabeled(200)
            train_complete_dataset = CIFARSIMLabeled(400)
            test_dataset = CIFARSIMLabeled(200)
        else:
            base_dataset = datasets.CIFAR10(file_path, train=True)

            train_labeled_idxs, train_unlabeled_idxs = label_index_split(
                base_dataset.targets, int(n_labeled / num_classes), num_classes
            )
            train_labeled_dataset = CIFAR10Labeled(
                file_path, train_labeled_idxs, train=True, transform=transforms_
            )
            train_unlabeled_dataset = CIFAR10Unlabeled(
                file_path, train_unlabeled_idxs, train=True, transform=transforms_
            )
            train_complete_dataset = CIFAR10Labeled(
                file_path, None, train=True, transform=transforms_
            )
            test_dataset = CIFAR10Labeled(
                file_path, train=False, transform=transforms_, download=True
            )
            print(
                "#Labeled:",
                len(train_labeled_idxs),
                "#Unlabeled:",
                len(train_unlabeled_idxs),
            )

        labeled_trainloader = torch.utils.data.DataLoader(
            train_labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        unlabeled_trainloader = torch.utils.data.DataLoader(
            train_unlabeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        dataset_bs = batch_size * 10
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=dataset_bs, shuffle=False, num_workers=0
        )
        train_complete_trainloader = torch.utils.data.DataLoader(
            train_complete_dataset,
            batch_size=dataset_bs,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        return (
            labeled_trainloader,
            unlabeled_trainloader,
            test_loader,
            train_complete_trainloader,
        )

    return prepare_data


def correct_counter(output, target, batch_size, topk=(1, 5)):
    tensor_target = torch.Tensor(target)
    dataset = torch.utils.data.TensorDataset(tensor_target)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    correct_counts = [0] * len(topk)
    for idx, tt in enumerate(dataloader):
        for i, k in enumerate(topk):
            _, pred = output[idx].topk(k, 1, True, True)
            correct_k = torch.eq(pred, tt[0].view(-1, 1)).sum().float().item()
            correct_counts[i] += correct_k

    print("correct_counts: ", correct_counts)
    return correct_counts


def do_test_sl_and_lia(config, alice, bob):
    device_y = bob

    tmp_dir = tempfile.TemporaryDirectory()
    lia_path = tmp_dir.name
    import logging

    logging.warning("lia_path: " + lia_path)

    # first, train a sl model and save model
    # prepare data
    # if you want to run this lia on CIFAR10, open this
    # data_file_path = lia_path + '/data_download'
    # train = True
    # train_dataset = datasets.CIFAR10(
    #     data_file_path, train, transform=transforms.ToTensor(), download=True
    # )
    # fake dataset, just for unittest
    train_dataset = CIFARSIMDataset(200)

    # dataloader transformed train_dataset, so we here call dataloader before we get data and label
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=False
    )

    model_save_path = lia_path + "/lia_model"

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
    optim_fn = optim_wrapper(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=5e-4)
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
        model_fn=TopModelForCifar10,
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

    def create_model(ema=False):
        bottom_model = BottomModelForCifar10()
        model = BottomModelPlus(bottom_model)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model(ema=False)
    ema_model = create_model(ema=True)

    data_buil = data_builder(batch_size=16, file_path=None)
    # for precision unittest
    # data_buil = data_builder(batch_size=16, file_path=data_file_path)

    lia_cb = LabelInferenceAttack(
        alice,
        model,
        ema_model,
        10,
        data_buil,
        attack_epochs=1,
        save_model_path=model_save_path,
        T=config["T"],
        alpha=config["alpha"],
        val_iteration=config["val_iteration"],
        k=config["k"],
        lr=config["lr"],
        ema_decay=config["ema_decay"],
        lambda_u=config["lambda_u"],
    )

    history = sl_model.fit(
        fed_data,
        label,
        validation_data=(fed_data, label),
        epochs=1,
        batch_size=128,
        shuffle=False,
        random_seed=1234,
        dataset_builder=None,
        callbacks=[lia_cb],
    )
    print(history)

    pred_bs = 128
    result = sl_model.predict(fed_data, batch_size=pred_bs, verbose=1)
    cor_count = bob(correct_counter)(result, label, batch_size=pred_bs, topk=(1, 4))
    sf.wait(cor_count)
    return lia_cb.get_attack_metrics()


def test_sl_and_lia(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    do_test_sl_and_lia(
        {
            "T": 0.8,
            "alpha": 0.75,
            "val_iteration": 1024,
            "k": 4,
            "lr": 2e-3,
            "ema_decay": 0.999,
            "lambda_u": 50,
        },
        alice,
        bob,
    )
