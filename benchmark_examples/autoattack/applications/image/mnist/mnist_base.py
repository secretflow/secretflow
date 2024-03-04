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

from abc import ABC
from typing import Dict, List, Tuple

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset

from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.utils.dataset_utils import sample_ndarray
from secretflow import reveal
from secretflow.utils.simulation.datasets import load_mnist


class MnistBase(ApplicationBase, ABC):
    def __init__(
        self,
        config,
        alice,
        bob,
        epoch=5,
        train_batch_size=128,
        hidden_size=612,
        dnn_fuse_units_size=None,
    ):
        super().__init__(
            config,
            alice,
            bob,
            device_y=bob,
            total_fea_nums=4000,
            alice_fea_nums=2000,
            num_classes=10,
            epoch=epoch,
            train_batch_size=train_batch_size,
            hidden_size=hidden_size,
            dnn_fuse_units_size=dnn_fuse_units_size,
        )

    def prepare_data(self, parts=None, is_torch=True, normalized_x=True):
        if parts is None:
            parts = {self.alice: (0, 14), self.bob: (14, 28)}
        (train_data, train_label), (test_data, test_label) = load_mnist(
            parts=parts,
            is_torch=is_torch,
            normalized_x=normalized_x,
            axis=3,
        )

        self.train_data = train_data.astype(np.float32)
        self.train_label = train_label
        self.test_data = test_data.astype(np.float32)
        self.test_label = test_label
        self.plain_alice_train_data = reveal(self.train_data.partitions[self.alice])
        self.plain_bob_train_data = reveal(self.train_data.partitions[self.bob])
        self.plain_train_label = reveal(self.train_label.partitions[self.bob])
        self.plain_test_label = reveal(self.test_label.partitions[self.bob])

    def lia_auxiliary_data_builder(self, batch_size=16, file_path=None):
        train = reveal(self.train_data.partitions[self.bob])
        tr_label = reveal(self.train_label.partitions[self.bob])
        test = reveal(self.test_data.partitions[self.bob])
        tst_label = reveal(self.test_label.partitions[self.bob])

        def split_some_data(df, label):
            sample_idx = np.random.choice(
                np.array([i for i in range(len(df))]), size=50, replace=False
            )
            sample_df = df[sample_idx]
            sample_label = label[sample_idx]
            datasets = torch.utils.data.TensorDataset(
                torch.tensor(sample_df), torch.tensor(sample_label)
            )
            return DataLoader(
                datasets,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )

        def prepare_data():
            train_complete_trainloader = DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(train), torch.tensor(tr_label)
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            train_labeled_dataloader = split_some_data(train, tr_label)
            train_unlabeled_dataloader = split_some_data(
                train, np.full(tr_label.shape, -1)
            )
            test_loader = DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(test), torch.tensor(tst_label)
                ),
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
        alice_train = self.plain_alice_train_data
        bob_train = self.plain_bob_train_data
        alice_train = sample_ndarray(alice_train)
        bob_train = sample_ndarray(bob_train)
        batch_size = self.train_batch_size

        def _prepare_data():
            alice_dataset = TensorDataset(torch.tensor(alice_train))
            bob_dataset = TensorDataset(torch.tensor(bob_train))
            alice_dataloader = DataLoader(
                dataset=alice_dataset, shuffle=False, batch_size=batch_size
            )
            bob_dataloader = DataLoader(
                dataset=bob_dataset, shuffle=False, batch_size=batch_size
            )
            dataloader_dict = {'alice': alice_dataloader, 'bob': bob_dataloader}
            return dataloader_dict, dataloader_dict

        return _prepare_data

    def fia_victim_mean_attr(self):
        alice_train = self.plain_alice_train_data.reshape(
            (self.plain_alice_train_data.shape[0], -1)
        )

        return sample_ndarray(alice_train).mean(axis=0)

    def fia_victim_input_shape(self):
        return list(self.plain_alice_train_data.shape[1:])

    def fia_attack_input_shape(self):
        return list(self.plain_bob_train_data.shape[1:])

    def fia_victim_model_dict(self, victim_model_save_path):
        return {self.device_f: [self.create_base_model_alice(), victim_model_save_path]}

    def replay_auxiliary_attack_configs(
        self, target_nums: int = 15
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        target_class = 8
        poison_class = 1
        target_indexes = np.where(np.array(self.plain_train_label) == target_class)[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)

        eval_indexes = np.where(np.array(self.plain_test_label) == poison_class)[0]
        eval_poison_set = np.random.choice(eval_indexes, 100, replace=False)
        return target_class, target_set, eval_poison_set

    def replace_auxiliary_attack_configs(self, target_nums: int = 15):
        target_class = 8
        target_indexes = np.where(np.array(self.plain_train_label) == target_class)[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)

        train_poison_set = np.random.choice(
            range(len(self.plain_train_label)), 100, replace=False
        )

        train_poison_np = np.stack(self.plain_alice_train_data[train_poison_set])

        eval_poison_set = np.random.choice(
            range(len(self.plain_test_label)), 100, replace=False
        )
        return (
            target_class,
            target_set,
            train_poison_set,
            train_poison_np,
            eval_poison_set,
        )

    def resources_consumes(self) -> List[Dict]:
        return [
            {'alice': 0.5, 'CPU': 0.5, 'GPU': 0.005, 'gpu_mem': 6 * 1024 * 1024 * 1024},
            {'bob': 0.5, 'CPU': 0.5, 'GPU': 0.005, 'gpu_mem': 6 * 1024 * 1024 * 1024},
        ]
