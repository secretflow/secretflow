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
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.utils.dataset_utils import sample_ndarray
from secretflow import reveal

from .data_utils import CIFAR10Labeled, CIFAR10Unlabeled, label_index_split


class Cifar10ApplicationBase(ApplicationBase, ABC):
    def __init__(
        self,
        config,
        alice,
        bob,
        epoch=20,
        train_batch_size=128,
        hidden_size=10,
        dnn_fuse_units_size=None,
    ):
        super().__init__(
            config,
            alice,
            bob,
            device_y=bob,
            total_fea_nums=32 * 32 * 3,
            alice_fea_nums=32 * 16 * 3,
            num_classes=10,
            epoch=epoch,
            train_batch_size=train_batch_size,
            hidden_size=hidden_size,
            dnn_fuse_units_size=dnn_fuse_units_size,
        )
        self.plain_alice_train_data: np.ndarray
        self.plain_bob_train_data: np.ndarray
        self.plain_train_label: np.ndarray

    def prepare_data(self):
        from secretflow.utils.simulation import datasets

        (self.train_data, self.train_label), (
            self.test_data,
            self.test_label,
        ) = datasets.load_cifar10(
            [self.alice, self.bob],
        )
        self.plain_alice_train_data = reveal(self.train_data.partitions[self.alice])
        self.plain_bob_train_data = reveal(self.train_data.partitions[self.bob])
        self.plain_train_label = reveal(self.train_label)
        self.plain_test_label = reveal(self.test_label)

    def alice_feature_nums_range(self) -> list:
        return [32 * 16 * 3]

    def lia_auxiliary_data_builder(
        self, batch_size=16, file_path="~/.secretflow/datasets/cifar10"
    ):
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

    def fia_auxiliary_data_builder(self):
        def _prepare_data():
            alice_train = self.plain_alice_train_data
            bob_train = self.plain_bob_train_data
            alice_train = sample_ndarray(alice_train)
            bob_train = sample_ndarray(bob_train)

            alice_dataset = TensorDataset(torch.tensor(alice_train))
            bob_dataset = TensorDataset(torch.tensor(bob_train))
            alice_dataloader = DataLoader(
                dataset=alice_dataset, shuffle=False, batch_size=self.train_batch_size
            )
            bob_dataloader = DataLoader(
                dataset=bob_dataset, shuffle=False, batch_size=self.train_batch_size
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
        # use 1 gpu per trail.
        return [
            {'alice': 0.5, 'CPU': 0.5, 'GPU': 0.005, 'gpu_mem': 6 * 1024 * 1024 * 1024},
            {'bob': 0.5, 'CPU': 0.5, 'GPU': 0.005, 'gpu_mem': 6 * 1024 * 1024 * 1024},
        ]
