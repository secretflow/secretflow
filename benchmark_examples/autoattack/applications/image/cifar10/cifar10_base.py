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
from typing import Dict

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
    DatasetType,
    InputMode,
)


class Cifar10ApplicationBase(ApplicationBase, ABC):
    def __init__(
        self,
        alice,
        bob,
        epoch=10,
        train_batch_size=128,
        hidden_size=10,
        dnn_fuse_units_size=None,
    ):
        super().__init__(
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

    def dataset_name(self):
        return 'cifar10'

    def prepare_data(self):
        from secretflow.utils.simulation import datasets

        (train_data, train_label), (
            test_data,
            test_label,
        ) = datasets.load_cifar10(
            [self.alice, self.bob],
        )

        if global_config.is_simple_test():
            sample_nums = 4000
            train_data = train_data[0:sample_nums]
            train_label = train_label.device(lambda df: df[0:sample_nums])(train_label)
            test_data = test_data[0:sample_nums]
            test_label = test_label.device(lambda df: df[0:sample_nums])(test_label)
        return train_data, train_label, test_data, test_label

    def tune_metrics(self) -> Dict[str, str]:
        return {
            "train_MulticlassAccuracy": "max",
            "train_MulticlassPrecision": "max",
            "train_MulticlassAUROC": "max",
            "val_MulticlassAccuracy": "max",
            "val_MulticlassPrecision": "max",
            "val_MulticlassAUROC": "max",
        }

    def classfication_type(self) -> ClassficationType:
        return ClassficationType.MULTICLASS

    def base_input_mode(self) -> InputMode:
        return InputMode.SINGLE

    def dataset_type(self) -> DatasetType:
        return DatasetType.IMAGE
