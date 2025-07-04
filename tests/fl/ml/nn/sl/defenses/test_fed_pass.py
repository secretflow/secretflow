# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, Precision

from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper


class TabDatasetLeft(Dataset):
    def __init__(self, data_num):
        self.data = torch.randn(data_num, 128)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data


class TabDatasetRight(Dataset):
    def __init__(self, data_num, return_y):
        self.data = torch.randn(data_num, 128)
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
            dataset = TabDatasetLeft(data_num)
        else:
            dataset = TabDatasetRight(data_num, return_y)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return dataloader

    return dataset_builder


def create_model_def(use_passport=False):
    input_dim = 128
    loss_fn = nn.CrossEntropyLoss
    optim_fn = optim_wrapper(torch.optim.Adam, lr=1e-3)

    base_model = TorchModel(
        model_fn=DnnBase,
        loss_fn=nn.BCELoss,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(Accuracy, task="binary"),
            metric_wrapper(Precision, task="binary"),
            metric_wrapper(AUROC, task="binary"),
        ],
        input_dims=[input_dim],
        dnn_units_size=[64, 10],
        use_passport=use_passport,
    )

    fuse_model = TorchModel(
        model_fn=DnnFuse,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=10, average="micro"
            ),
            metric_wrapper(AUROC, task="multiclass", num_classes=10),
        ],
        input_dims=[10, 10],
        dnn_units_size=[20, 10],
        output_func=nn.Softmax,
        use_passport=use_passport,
    )
    return base_model, fuse_model


def test_fed_pass(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    device_y = bob

    data_num = 40
    batch_size = 8

    base_model, fuse_model = create_model_def(use_passport=False)
    base_model_fedpass, fuse_model_fedpass = create_model_def(use_passport=True)

    base_model_dict = {
        alice: base_model,
        bob: base_model,
    }
    base_model_dict_fedpass = {
        alice: base_model_fedpass,
        bob: base_model_fedpass,
    }

    dataset_builder_dict = {
        alice: create_dataset_builder(
            data_num=data_num,
            batch_size=batch_size,
            is_left=True,
        ),
        bob: create_dataset_builder(
            data_num=data_num,
            batch_size=batch_size,
            is_left=False,
        ),
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=device_y,
        model_fuse=fuse_model,
        simulation=True,
        random_seed=1234,
        backend="torch",
        strategy="split_nn",
    )

    sl_model_fedpass = SLModel(
        base_model_dict=base_model_dict_fedpass,
        device_y=device_y,
        model_fuse=fuse_model_fedpass,
        simulation=True,
        random_seed=1234,
        backend="torch",
        strategy="split_nn",
    )

    train_data = torch.rand(data_num, 256)

    train_label = np.random.randint(0, 10, size=(data_num,))
    data = FedNdarray(
        partitions={
            alice: alice(lambda x: x[:, :128])(train_data),
            bob: bob(lambda x: x[:, 128:])(train_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label = bob(lambda x: x)(train_label)

    history = sl_model_fedpass.fit(
        data,
        label,
        epochs=1,
        batch_size=batch_size,
        shuffle=False,
        random_seed=1234,
        dataset_builder=dataset_builder_dict,
    )
    logging.info(history)

    pred_dataset_builder_dict = {
        alice: create_dataset_builder(
            data_num=data_num,
            batch_size=batch_size,
            is_left=True,
        ),
        bob: create_dataset_builder(
            data_num=data_num,
            batch_size=batch_size,
            is_left=False,
            return_y=False,
        ),
    }

    preds_fedpass = sl_model_fedpass.predict(
        data,
        batch_size=128,
        dataset_builder=pred_dataset_builder_dict,
    )

    pred = sl_model.predict(
        data,
        batch_size=128,
        dataset_builder=pred_dataset_builder_dict,
    )
    assert len(preds_fedpass) == len(pred)
