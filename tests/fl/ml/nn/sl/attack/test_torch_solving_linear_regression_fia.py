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

import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import AUROC

import secretflow as sf
from secretflow.data.split import train_test_split
from secretflow.utils.simulation.data.dataframe import create_df
from secretflow_fl.ml.nn import SLModel
from secretflow_fl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow_fl.ml.nn.sl.agglayer.agg_method import Concat
from secretflow_fl.ml.nn.sl.attacks.solving_linear_regression_fia_torch import (
    SolvingLinearRegressionAttack,
)


class BankModelAlice(nn.Module):
    def __init__(self):
        super(BankModelAlice, self).__init__()
        self.fc1 = nn.Linear(
            8, 30
        )  # Input layer: 20 features, first hidden layer: 60 neurons

    def forward(self, x):
        x = self.fc1(x)

        return x

    def output_num(self):
        return 1


class BankModelBob(nn.Module):
    def __init__(self):
        super(BankModelBob, self).__init__()
        self.fc1 = nn.Linear(
            12, 30
        )  # Input layer: 20 features, first hidden layer: 60 neurons

    def forward(self, x):
        x = self.fc1(x)

        return x

    def output_num(self):
        return 1


class BankModelFuse(nn.Module):
    def __init__(self):
        super(BankModelFuse, self).__init__()
        # self.fc1 = nn.Linear(12, 60)  # Input layer: 20 features, first hidden layer: 60 neurons
        self.fc2 = nn.Linear(60, 60)  # Second hidden layer: 30 neurons
        self.fc3 = nn.Linear(60, 10)  # Third hidden layer: 10 neurons
        self.fc4 = nn.Linear(10, 1)  # Output layer: 2 classes

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        x = torch.cat(x, dim=1)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


def data_builder(data, label, batch_size):
    def prepare_data():
        print("prepare_data num: ", data.shape)
        alice_data = data[:, :28]
        bob_data = data[:, 28:]

        alice_dataset = TensorDataset(torch.tensor(alice_data))
        alice_dataloader = DataLoader(
            dataset=alice_dataset,
            shuffle=False,
            batch_size=batch_size,
        )

        bob_dataset = TensorDataset(torch.tensor(bob_data))
        bob_dataloader = DataLoader(
            dataset=bob_dataset,
            shuffle=False,
            batch_size=batch_size,
        )

        dataloader_dict = {"alice": alice_dataloader, "bob": bob_dataloader}
        return dataloader_dict, dataloader_dict

    return prepare_data


def do_test_sl_and_fia(alice, bob):
    # fake data
    size = 200
    arr = np.random.rand(size, 20)

    # Change the third column to randomly generated 0s and 1s
    arr[:, 3] = np.random.randint(0, 2, size=size)
    # fake column
    alice_feature = [
        "age",
        "previous",
        "emp.var.rate",
        "contact",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    bob_feature = [
        "job",
        "default",
        "housing",
        "loan",
        "day_of_week",
        "marital",
        "poutcome",
        "education",
        "month",
        "duration",
        "campaign",
        "pdays",
    ]

    all_feature = alice_feature + bob_feature

    df = pd.DataFrame(arr, columns=all_feature)
    X = df.astype(np.float32)

    print(X.head(5))

    y = np.random.randint(0, 2, size=size)
    y = pd.DataFrame(y, columns=["y"]).astype(np.float32)

    data = create_df(
        source=X,
        parts={
            alice: (0, 8),
            bob: (8, 20),
        },
        axis=1,
        shuffle=False,
        aggregator=None,
        comparator=None,
    )
    label = create_df(
        source=y,
        parts={bob: (0, 1)},
        axis=1,
        shuffle=False,
        aggregator=None,
        comparator=None,
    ).astype(np.float32)
    random_state = 1
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=random_state
    )
    train_label, test_label = train_test_split(
        label, train_size=0.8, random_state=random_state
    )

    batch_size = 128
    loss_fn = nn.BCELoss
    optim_fn = optim_wrapper(torch.optim.Adam, lr=1e-3)
    alice_model = TorchModel(
        model_fn=BankModelAlice,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[metric_wrapper(AUROC, task="binary")],
    )
    bob_model = TorchModel(
        model_fn=BankModelBob,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[metric_wrapper(AUROC, task="binary")],
    )

    fuse_model = TorchModel(
        model_fn=BankModelFuse,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[metric_wrapper(AUROC, task="binary")],
    )

    base_model_dict = {
        alice: alice_model,
        bob: bob_model,
    }
    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=bob,
        model_fuse=fuse_model,
        dp_strategy_dict=None,
        compressor=None,
        simulation=True,
        random_seed=1234,
        backend="torch",
        strategy="split_nn",
    )

    fia_callback = SolvingLinearRegressionAttack(
        attack_party=bob, victim_party=alice, r=9, targets_columns=[3, 4, 5]
    )
    sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=1,
        batch_size=batch_size,
        shuffle=False,
        random_seed=1234,
        dataset_builder=None,
        callbacks=[fia_callback],
    )
    metrics = fia_callback.get_attack_metrics()
    print(metrics)
    return metrics


def test_sl_and_fia(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    do_test_sl_and_fia(alice, bob)
