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

import logging
from abc import ABC
from typing import List, Optional, Union

import torch
from torchvision import datasets, transforms

from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.callbacks.callback import Callback

from .data_utils import CIFAR10Labeled, CIFAR10Unlabeled, label_index_split


class Cifar10TrainBase(TrainBase, ABC):
    def __init__(self, config, alice, bob, epoch=1, train_batch_size=128):
        super().__init__(
            config, alice, bob, bob, 10, epoch=epoch, train_batch_size=train_batch_size
        )

    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        sl_model = SLModel(
            base_model_dict={
                self.alice: self.alice_base_model,
                self.bob: self.bob_base_model,
            },
            device_y=self.device_y,
            model_fuse=self.fuse_model,
            simulation=True,
            random_seed=1234,
            backend='torch',
            strategy='split_nn',
        )
        history = sl_model.fit(
            x=self.train_data,
            y=self.train_label,
            validation_data=(self.test_data, self.test_label),
            epochs=self.epoch,
            batch_size=self.train_batch_size,
            shuffle=False,
            random_seed=1234,
            dataset_builder=None,
            callbacks=callbacks,
        )
        logging.warning(history)

    def _prepare_data(self):
        from secretflow.utils.simulation import datasets

        (train_data, train_label), (test_data, test_label) = datasets.load_cifar10(
            [self.alice, self.bob],
        )

        return train_data, train_label, test_data, test_label

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
