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
from typing import Dict, List, Tuple

import numpy as np

from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.global_config import is_simple_test
from secretflow import reveal
from secretflow.data.split import train_test_split
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
        config,
        alice,
        bob,
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
            config,
            alice,
            bob,
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
        names = list(all_features.keys())
        self.alice_fea_classes = {
            names[i]: all_features[names[i]] for i in range(self.alice_fea_nums)
        }
        self.bob_fea_classes = {
            names[i + self.alice_fea_nums]: all_features[names[i + self.alice_fea_nums]]
            for i in range(self.bob_fea_nums)
        }
        self.plain_alice_train_data = None
        self.plain_bob_train_data = None
        self.plain_train_label = None
        self.plain_test_label = None

    def prepare_data(self):
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
            data[col] = encoder.fit_transform(data[col])
        label = encoder.fit_transform(label)
        random_state = 1234
        self.train_data, self.test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        self.train_label, self.test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )
        self.plain_alice_train_data = reveal(
            self.train_data.partitions[self.alice].data
        )
        self.plain_bob_train_data = reveal(self.train_data.partitions[self.bob].data)
        self.plain_train_label = reveal(self.train_label.partitions[self.alice].data)
        self.plain_test_label = reveal(self.test_label.partitions[self.alice].data)

    def alice_feature_nums_range(self) -> list:
        # support 1-16
        return [1, 5, 9, 10, 15]

    def hidden_size_range(self) -> list:
        return [32, 64]

    def exploit_label_counts(self) -> Tuple[int, int]:
        neg, pos = np.bincount(self.plain_train_label['y'])
        return neg, pos

    def resources_consumes(self) -> List[Dict]:
        return [
            {'alice': 0.5, 'CPU': 0.5, 'GPU': 0.001, 'gpu_mem': 500 * 1024 * 1024},
            {'bob': 0.5, 'CPU': 0.5, 'GPU': 0.001, 'gpu_mem': 500 * 1024 * 1024},
        ]
