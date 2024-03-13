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
import sys

PROJ_DIR = "/home/kyzhang/myfile/sf/github/Norm_Attack"
sys.path.append(os.path.join(PROJ_DIR))


import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils.utils import NumpyDataset


def prepareFederatedMNISTDataloaders(
    client_num=2,
    local_label_num=2,
    local_data_num=20,
    batch_size=1,
    test_batch_size=16,
    path="MNIST/.",
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
    seed=0,
    return_idx=False,
):
    np.random.seed(seed)
    random.seed(seed)

    at_t_dataset_train = torchvision.datasets.MNIST(
        root=path, train=True, download=download
    )
    at_t_dataset_test = torchvision.datasets.MNIST(
        root=path, train=False, download=download
    )

    X = at_t_dataset_train.train_data.numpy()
    y = at_t_dataset_train.train_labels.numpy()

    test_set = NumpyDataset(
        at_t_dataset_test.test_data.numpy(),
        at_t_dataset_test.test_labels.numpy(),
        transform=transform,
        return_idx=return_idx,
    )
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=True, num_workers=0
    )

    trainloaders = []
    train_sizes = []
    idx_used = []
    for c in range(client_num):
        assigned_labels = random.sample(range(10), local_label_num)
        print(f"the labels that client_id={c} has are: ", assigned_labels)
        idx = np.concatenate([np.where(y == al)[0] for al in assigned_labels])
        assigned_idx = random.sample(list(set(idx) - set(idx_used)), local_data_num)

        temp_trainset = NumpyDataset(
            X[assigned_idx], y[assigned_idx], transform=transform, return_idx=return_idx
        )
        temp_trainloader = torch.utils.data.DataLoader(
            temp_trainset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        trainloaders.append(temp_trainloader)
        train_sizes.append(len(temp_trainset))

        idx_used += assigned_idx

    return X, y, trainloaders, testloader, train_sizes, idx_used
