# Copyright 2023 Ant Group Co., Ltd.
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

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack.applications.table.bank.bank_base import BankBase
from benchmark_examples.autoattack.utils.dataset_utils import (
    create_custom_dataset_builder,
)
from secretflow import reveal
from secretflow.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper


class AliceDataset(Dataset):
    def __init__(self, x, loss='binary'):
        df = x[0]
        label = x[1]
        for feat in df.columns:
            mms = MinMaxScaler()
            df[feat] = mms.fit_transform(df[[feat]])
        if loss == "binary":
            self.label = torch.tensor(label.values.astype(np.float32))
        else:
            self.label = torch.tensor(label['y'].values)

        self.tensor = torch.tensor(df.values.astype(np.float32))

    def __getitem__(self, index):
        return self.tensor[index], self.label[index]

    def __len__(self):
        return self.tensor.size(0)


# CMDataset = AliceDataset


class BobDataset(Dataset):
    def __init__(self, x, tuple_type=False):
        df = x[0]
        for feat in df.columns:
            mms = MinMaxScaler()
            df[feat] = mms.fit_transform(df[[feat]])
        self.tensor = torch.tensor(df.values.astype(np.float32))
        self.tuple_type = tuple_type

    def __getitem__(self, index):
        return tuple((self.tensor[index],)) if self.tuple_type else self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)


class BankDnn(BankBase):
    def __init__(
        self,
        config,
        alice,
        bob,
        hidden_size=64,
    ):
        super().__init__(
            config,
            alice,
            bob,
            hidden_size=hidden_size,
            dnn_base_units_size_alice=[100, hidden_size],
            dnn_base_units_size_bob=[100, hidden_size],
            dnn_fuse_units_size=[1],
        )

    def create_base_model(self, input_dim, dnn_units_size):
        model = TorchModel(
            model_fn=DnnBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(Precision, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[input_dim],
            dnn_units_size=dnn_units_size,
        )
        return model

    def create_base_model_alice(self):
        return self.create_base_model(
            self.alice_fea_nums, self.dnn_base_units_size_alice
        )

    def create_base_model_bob(self):
        return self.create_base_model(self.bob_fea_nums, self.dnn_base_units_size_bob)

    def create_fuse_model(self):
        return TorchModel(
            model_fn=DnnFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(Precision, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=self.dnn_fuse_units_size,
            output_func=nn.Sigmoid,
        )

    def create_dataset_builder_alice(self, *args, **kwargs):
        return create_custom_dataset_builder(AliceDataset, self.train_batch_size)

    def create_dataset_builder_bob(self, *args, **kwargs):
        return create_custom_dataset_builder(BobDataset, self.train_batch_size)

    def create_predict_dataset_builder_alice(
        self, *args, **kwargs
    ) -> Optional[Callable]:
        return create_custom_dataset_builder(BobDataset, self.train_batch_size)

    def create_predict_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return create_custom_dataset_builder(BobDataset, self.train_batch_size)

    def support_attacks(self):
        return ['norm', 'lia', 'fia', 'replay', 'replace', 'exploit']

    def dnn_base_units_size_range_alice(self) -> Optional[list]:
        return [
            [128, -1],
            [-1],
        ]

    def dnn_base_units_size_range_bob(self) -> Optional[list]:
        # since alice = bob
        return None

    def dnn_fuse_units_size_range(self) -> Optional[list]:
        return [[1], [128, 1]]

    def lia_auxiliary_model(self, ema=False):
        from benchmark_examples.autoattack.attacks.lia import BottomModelPlus

        bottom_model = DnnBase(
            input_dims=[self.alice_fea_nums],
            dnn_units_size=self.dnn_base_units_size_alice,
        )
        model = BottomModelPlus(
            bottom_model,
            size_bottom_out=self.dnn_base_units_size_alice[-1],
            num_classes=2,
        )

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    def lia_auxiliary_data_builder(self, batch_size=16, file_path=None):
        train = self.plain_bob_train_data
        tr_label = self.plain_train_label
        test = reveal(self.test_data.partitions[self.bob].data)
        tst_label = reveal(self.test_label.partitions[self.alice].data)

        def split_some_data(df, label):
            df['y'] = label['y']
            df = df.sample(n=50, random_state=42)
            label = df[['y']]
            df = df.drop(columns=['y'])
            datasets = AliceDataset([df, label], loss='multi-class')
            return DataLoader(
                datasets,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )

        def prepare_data():
            train_complete_trainloader = DataLoader(
                AliceDataset([train, tr_label], loss='multi-class'),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            train_labeled_dataloader = split_some_data(train, tr_label)
            tr_label['y'] = -1
            train_unlabeled_dataloader = split_some_data(train, tr_label)
            test_loader = DataLoader(
                AliceDataset([test, tst_label], loss='multi-class'),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            return (
                train_labeled_dataloader,
                train_unlabeled_dataloader,
                test_loader,
                train_complete_trainloader,
            )

        return prepare_data

    def fia_auxiliary_data_builder(self):
        alice_train = self.plain_alice_train_data.sample(frac=0.4, random_state=42)
        bob_train = self.plain_bob_train_data.sample(frac=0.4, random_state=42)
        train_batch_size = self.train_batch_size

        def _prepare_data():
            alice_dataset = BobDataset([alice_train], tuple_type=True)
            bob_dataset = BobDataset([bob_train], tuple_type=True)
            alice_dataloader = DataLoader(
                dataset=alice_dataset, shuffle=False, batch_size=train_batch_size
            )
            bob_dataloader = DataLoader(
                dataset=bob_dataset, shuffle=False, batch_size=train_batch_size
            )
            dataloader_dict = {'alice': alice_dataloader, 'bob': bob_dataloader}
            return dataloader_dict, dataloader_dict

        return _prepare_data

    def fia_victim_mean_attr(self):
        df = self.plain_bob_train_data.sample(frac=0.4, random_state=42)
        mms = MinMaxScaler()
        for feat in df.columns:
            df[feat] = mms.fit_transform(df[[feat]])
        return df.values.mean(axis=0)

    def fia_victim_model_dict(self, victim_model_save_path):
        return {self.device_f: [self.create_base_model_bob(), victim_model_save_path]}

    def fia_victim_input_shape(self):
        return list(self.plain_bob_train_data.shape[1:])

    def fia_attack_input_shape(self):
        return list(self.plain_alice_train_data.shape[1:])

    def replay_auxiliary_attack_configs(
        self, target_nums: int = 15
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        target_class = 1
        poison_class = 0
        target_indexes = np.where(np.array(self.plain_train_label) == target_class)[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)
        eval_indexes = np.where(np.array(self.plain_test_label) == poison_class)[0]
        eval_poison_set = np.random.choice(eval_indexes, 100, replace=False)
        return target_class, target_set, eval_poison_set

    def replace_auxiliary_attack_configs(self, target_nums: int = 15):
        plain_train_label = np.array(self.plain_train_label)
        plain_test_label = np.array(self.plain_test_label)
        plain_bob_train_data = self.plain_bob_train_data.copy()
        mms = MinMaxScaler()
        for feat in plain_bob_train_data.columns:
            plain_bob_train_data[feat] = mms.fit_transform(plain_bob_train_data[[feat]])
        plain_bob_train_data = plain_bob_train_data.values
        target_class = 1
        target_indexes = np.where(plain_train_label == target_class)[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)
        train_poison_set = np.random.choice(
            range(len(plain_train_label)), 100, replace=False
        )
        train_poison_np = np.stack(plain_bob_train_data[train_poison_set])
        eval_poison_set = np.random.choice(
            range(len(plain_test_label)), 100, replace=False
        )
        return (
            target_class,
            target_set,
            train_poison_set,
            train_poison_np,
            eval_poison_set,
        )
