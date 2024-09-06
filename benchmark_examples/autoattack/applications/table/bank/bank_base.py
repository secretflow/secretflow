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

from abc import ABC
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
    DatasetType,
)
from benchmark_examples.autoattack.global_config import is_simple_test
from benchmark_examples.autoattack.utils.resources import ResourceDict, ResourcesPack
from secretflow.data.vertical import VDataFrame
from secretflow.preprocessing import LabelEncoder
from secretflow.utils.simulation.datasets import load_bank_marketing

all_features = OrderedDict(
    {
        'age': 100,
        'job': 12,
        'marital': 3,
        'education': 4,
        # default split ----
        'default': 2,
        'balance': 7168,  # 2353 for small dataset
        'housing': 2,
        'loan': 2,
        'contact': 3,
        'day': 31,
        'month': 12,
        'duration': 1573,  # 875,
        'campaign': 48,  # 32,
        'pdays': 559,  # 292,
        'previous': 41,  # 24,
        'poutcome': 4,
    }
)


class BankBase(ApplicationBase, ABC):
    def __init__(
        self,
        alice,
        bob,
        has_custom_dataset=False,
        epoch=5,
        train_batch_size=128,
        hidden_size=64,
        alice_fea_nums=9,
        dnn_base_units_size_alice=None,
        dnn_base_units_size_bob=None,
        dnn_fuse_units_size=None,
        dnn_embedding_dim=None,
        deepfm_embedding_dim=None,
    ):
        super().__init__(
            alice,
            bob,
            has_custom_dataset=has_custom_dataset,
            device_y=alice,
            total_fea_nums=16,
            alice_fea_nums=alice_fea_nums,
            num_classes=2,
            epoch=epoch,
            train_batch_size=train_batch_size,
            hidden_size=hidden_size,
            dnn_base_units_size_alice=dnn_base_units_size_alice,
            dnn_base_units_size_bob=dnn_base_units_size_bob,
            dnn_fuse_units_size=dnn_fuse_units_size,
            dnn_embedding_dim=dnn_embedding_dim,
            deepfm_embedding_dim=deepfm_embedding_dim,
        )
        self.alice_fea_classes = None
        self.bob_fea_classes = None
        self.train_dataset_len = 36168
        self.test_dataset_len = 9043
        if global_config.is_simple_test():
            self.train_dataset_len = 3616
            self.test_dataset_len = 905

    def dataset_name(self):
        return "bank"

    def set_config(self, config: Dict[str, str] | None):
        super().set_config(config)
        names = list(all_features.keys())
        self.alice_fea_classes = {
            names[i]: all_features[names[i]] for i in range(self.alice_fea_nums)
        }
        self.bob_fea_classes = {
            names[i + self.alice_fea_nums]: all_features[names[i + self.alice_fea_nums]]
            for i in range(self.bob_fea_nums)
        }

    def load_bank_data(self) -> Tuple[VDataFrame, VDataFrame]:
        """Since different implements have different processing on dataset
        overridd this method and preprocess data,
        use super() to get data.
        """
        data = load_bank_marketing(
            parts={
                self.alice: (0, self.alice_fea_nums),
                self.bob: (self.alice_fea_nums, 16),
            },
            full=False if is_simple_test() else True,
            axis=1,
        )
        label = load_bank_marketing(
            parts={self.alice: (16, 17)},
            full=False if is_simple_test() else True,
            axis=1,
        )
        encoder = LabelEncoder()
        for col in data.columns:
            data[col] = encoder.fit_transform(data[[col]])
        label = encoder.fit_transform(label['y'])
        return data.astype(np.float32), label.astype(np.float32)

    def classfication_type(self) -> ClassficationType:
        return ClassficationType.BINARY

    def dataset_type(self) -> DatasetType:
        return DatasetType.TABLE

    def resources_consumption(self) -> ResourcesPack:
        # 442MiB
        return (
            ResourcesPack()
            .with_debug_resources(
                ResourceDict(
                    gpu_mem=500 * 1024 * 1024, CPU=1, memory=1.2 * 1024 * 1024 * 1024
                )
            )
            .with_sim_resources(
                self.device_y.party,
                ResourceDict(
                    gpu_mem=500 * 1024 * 1024, CPU=1, memory=1.2 * 1024 * 1024 * 1024
                ),
            )
            .with_sim_resources(
                self.device_f.party,
                ResourceDict(
                    gpu_mem=400 * 1024 * 1024, CPU=1, memory=1.2 * 1024 * 1024 * 1024
                ),
            )
        )
