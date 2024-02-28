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

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset
from torchmetrics import AUROC, Accuracy

from benchmark_examples.autoattack.applications.table.bank.bank_base import BankBase
from benchmark_examples.autoattack.utils.dataset_utils import (
    create_custom_dataset_builder,
)
from secretflow.ml.nn.applications.sl_deepfm_torch import DeepFMBase, DeepFMFuse
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper


class AliceDataset(Dataset):
    def __init__(self, x):
        df = x[0]
        label = x[1]
        self.tensors = []
        for feat in df.columns:
            v = df[feat]
            v = v.reset_index(drop=True)
            self.tensors.append(torch.tensor(v))
        self.label = torch.tensor(label.values.astype(np.float32))

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), self.label[index]

    def __len__(self):
        return self.tensors[0].size(0)


class BobDataset(Dataset):
    def __init__(self, x):
        df = x[0]
        self.tensors = []
        for feas in df.columns:
            v = df[feas].reset_index(drop=True)
            self.tensors.append(torch.tensor(v))

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class BankDeepfm(BankBase):
    def __init__(self, config, alice, bob, hidden_size=64):
        super().__init__(
            config,
            alice,
            bob,
            hidden_size=hidden_size,
            dnn_base_units_size_alice=[100, hidden_size],
            dnn_base_units_size_bob=[100, hidden_size],
            dnn_fuse_units_size=[64],
            deepfm_embedding_dim=4,
        )

    def create_base_model_alice(self):
        model = TorchModel(
            model_fn=DeepFMBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[v for v in self.alice_fea_classes.values()],
            dnn_units_size=self.dnn_base_units_size_alice,
        )
        return model

    def create_base_model_bob(self):
        model = TorchModel(
            model_fn=DeepFMBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[v for v in self.bob_fea_classes.values()],
            dnn_units_size=self.dnn_base_units_size_bob,
            fm_embedding_dim=self.deepfm_embedding_dim,
        )
        return model

    def create_fuse_model(self):
        return TorchModel(
            model_fn=DeepFMFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=self.dnn_fuse_units_size,
        )

    def create_dataset_builder_alice(self):
        batch_size = self.train_batch_size
        return create_custom_dataset_builder(AliceDataset, batch_size)

    def create_dataset_builder_bob(self):
        batch_size = self.train_batch_size
        return create_custom_dataset_builder(BobDataset, batch_size)

    def create_predict_dataset_builder_alice(
        self, *args, **kwargs
    ) -> Optional[Callable]:
        return create_custom_dataset_builder(BobDataset, self.train_batch_size)

    def create_predict_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return create_custom_dataset_builder(BobDataset, self.train_batch_size)

    def support_attacks(self):
        return ['norm', "replay", "replace"]

    def dnn_base_units_size_range_alice(self):
        return [
            [128, -1],
            [128, 128, -1],
            [-1],
        ]

    def dnn_base_units_size_range_bob(self) -> Optional[List[List[int]]]:
        return None

    def dnn_fuse_units_size_range(self):
        return [[64], [64, 64], [64, 64, 64]]

    def deepfm_embedding_dim_range(self):
        return [8, 16]

    def replay_auxiliary_attack_configs(
        self, target_nums: int = 15
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        plain_train_label = self.plain_train_label
        plain_test_label = self.plain_test_label
        target_class = 1
        poison_class = 0
        target_indexes = np.where(np.array(plain_train_label) == target_class)[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)

        eval_indexes = np.where(np.array(plain_test_label) == poison_class)[0]
        eval_poison_set = np.random.choice(eval_indexes, 100, replace=False)
        return target_class, target_set, eval_poison_set

    def replace_auxiliary_attack_configs(self, target_nums: int = 15):
        plain_train_label = np.array(self.plain_train_label)
        plain_test_label = np.array(self.plain_test_label)
        target_class = 1
        target_indexes = np.where(plain_train_label == target_class)[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)

        train_poison_set = np.random.choice(
            range(len(plain_train_label)), 100, replace=False
        )
        plain_bob_train_data = self.plain_bob_train_data.copy()
        train_poison_data = []
        for feat in plain_bob_train_data.columns:
            train_poison_data.append(
                plain_bob_train_data[feat].values[train_poison_set]
            )

        train_poison_np = [np.stack(data) for data in train_poison_data]
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
