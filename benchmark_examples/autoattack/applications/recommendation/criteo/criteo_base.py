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
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.global_config import is_simple_test
from secretflow import reveal
from secretflow.data.split import train_test_split
from secretflow.utils.simulation import datasets

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
all_features = ['I' + str(i) for i in range(1, 14)] + [
    'C' + str(i) for i in range(1, 27)
]
sparse_classes = OrderedDict(
    {
        'C1': 971,
        'C2': 525,
        'C3': 151956,
        'C4': 67777,
        'C5': 222,
        'C6': 14,
        'C7': 9879,
        'C8': 456,
        'C9': 3,
        'C10': 21399,
        'C11': 4418,
        'C12': 132753,
        'C13': 3015,
        'C14': 26,
        'C15': 7569,
        'C16': 105768,
        'C17': 10,
        'C18': 3412,
        'C19': 1680,
        'C20': 4,
        'C21': 121067,
        'C22': 14,
        'C23': 15,
        'C24': 26889,
        'C25': 60,
        'C26': 20490,
    }
)


class AliceDataset(Dataset):
    def __init__(self, x, has_label=True):
        df = x[0].fillna('-1')
        self.has_label = has_label
        self.tensors = []
        my_sparse_cols = []
        has_dense = False
        for col in df.columns:
            if col in sparse_classes.keys():
                lbe = LabelEncoder()
                v = lbe.fit_transform(df[col].fillna('0'))
                my_sparse_cols.append(col)
                self.tensors.append(torch.tensor(v, dtype=torch.long))
            else:
                has_dense = True
        if has_dense:
            df = df.drop(columns=my_sparse_cols)
            df = df.fillna(0)
            mms = MinMaxScaler(feature_range=(0, 1))
            self.dense_array = mms.fit_transform(df).astype(np.float32)
            self.tensors.insert(0, torch.tensor(self.dense_array))
        if has_label:
            self.label_tensor = torch.tensor(x[1].values.astype(np.float32))

    def __getitem__(self, index):
        if self.has_label:
            return (
                tuple(tensor[index] for tensor in self.tensors),
                self.label_tensor[index],
            )
        else:
            return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

    def get_dense_array(self):
        return self.dense_array


class BobDataset(Dataset):
    def __init__(self, x):
        df = x[0].fillna('-1')
        self.tensors = []
        my_sparse_cols = []
        has_dense = False
        for col in df.columns:
            if col in sparse_classes.keys():
                lbe = LabelEncoder()
                v = lbe.fit_transform(df[col])
                my_sparse_cols.append(col)
                self.tensors.append(torch.tensor(v, dtype=torch.long))
            else:
                has_dense = True
        if has_dense:
            df = df.drop(columns=my_sparse_cols)
            df = df.fillna(0)
            mms = MinMaxScaler(feature_range=(0, 1))
            dense_array = mms.fit_transform(df).astype(np.float32)
            self.tensors.insert(0, torch.tensor(dense_array))

    def __getitem__(self, index):
        # 这里和tf实现有不同，tf使用了feature_column.categorical_column_with_identity
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class CriteoBase(ApplicationBase, ABC):
    def __init__(
        self,
        config,
        alice,
        bob,
        epoch=2,
        train_batch_size=64,
        hidden_size=64,
        alice_fea_nums=13,
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
            device_y=bob,
            total_fea_nums=39,
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
        self.alice_input_dims = []
        self.alice_sparse_indexes = None
        self.alice_dense_indexes = [0]
        if self.alice_fea_nums <= 13:
            self.alice_input_dims = [self.alice_fea_nums]
        else:
            self.alice_input_dims = [13] + [
                list(sparse_classes.values())[i]
                for i in range(self.alice_fea_nums - 13)
            ]
            self.alice_sparse_indexes = [
                i for i in range(1, len(self.alice_input_dims))
            ]
        self.bob_input_dims = []
        self.bob_sparse_indexes = None
        self.bob_dense_indexes = None
        if self.alice_fea_nums >= 13:
            self.bob_input_dims = [
                list(sparse_classes.values())[i]
                for i in range(self.alice_fea_nums - 13, len(sparse_classes))
            ]
            self.bob_sparse_indexes = [i for i in range(len(self.bob_input_dims))]
        else:
            self.bob_input_dims = [13 - self.alice_fea_nums] + list(
                sparse_classes.values()
            )
            self.bob_sparse_indexes = [i for i in range(1, len(self.bob_input_dims))]
            self.bob_dense_indexes = [0]
        self.plain_alice_train_data = None
        self.plain_bob_train_data = None
        self.plain_train_label = None
        self.plain_test_label = None

    def prepare_data(self):
        random_state = 1234
        num_samples = 1000 if is_simple_test() else 410000
        # need to read label, so + 1
        data = datasets.load_criteo(
            {
                self.alice: (1, self.alice_fea_nums + 1),
                self.bob: (self.alice_fea_nums + 1, 40),
            },
            num_samples=num_samples,
        )
        label = datasets.load_criteo({self.bob: (0, 1)}, num_samples=num_samples)
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
        self.plain_train_label = reveal(self.train_label.partitions[self.bob].data)
        self.plain_test_label = reveal(self.test_label.partitions[self.bob].data)

    def create_dataset_builder_alice(self):
        train_batch_size = self.train_batch_size

        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = AliceDataset(x, has_label=False)
            dataloader = torch_data.DataLoader(
                dataset=data_set, batch_size=train_batch_size
            )
            return dataloader

        return dataset_builder

    def create_dataset_builder_bob(self):
        train_batch_size = self.train_batch_size

        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = AliceDataset(x, has_label=True)
            dataloader = torch_data.DataLoader(
                dataset=data_set, batch_size=train_batch_size
            )
            return dataloader

        return dataset_builder

    def create_predict_dataset_builder_alice(
        self, *args, **kwargs
    ) -> Optional[Callable]:
        return self.create_dataset_builder_alice()

    def create_predict_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return self.create_dataset_builder_alice()

    def alice_feature_nums_range(self) -> list:
        # support range 1 - 37
        return [2, 5, 13, 18, 37]

    def fia_auxiliary_data_builder(self):
        alice_train = self.plain_alice_train_data.sample(frac=0.4, random_state=42)
        bob_train = self.plain_bob_train_data.sample(frac=0.4, random_state=42)
        train_batch_size = self.train_batch_size

        def _prepare_data():
            alice_dataset = AliceDataset([alice_train], has_label=False)
            bob_dataset = AliceDataset([bob_train], has_label=False)
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
        train_sample = self.plain_alice_train_data.sample(frac=0.4, random_state=42)
        dataset = AliceDataset([train_sample], has_label=False)
        return dataset.get_dense_array()

    def fia_victim_model_dict(self, victim_model_save_path):
        return {self.device_f: [self.create_base_model_alice(), victim_model_save_path]}

    def replay_auxiliary_attack_configs(
        self, target_nums: int = 15
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        plain_train_label = LabelEncoder().fit_transform(self.plain_train_label)
        plain_test_label = LabelEncoder().fit_transform(self.plain_test_label)
        target_class = 1
        poison_class = 0
        target_indexes = np.where(np.array(plain_train_label) == target_class)[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)
        eval_indexes = np.where(np.array(plain_test_label) == poison_class)[0]
        eval_poison_set = np.random.choice(eval_indexes, 100, replace=False)
        return target_class, target_set, eval_poison_set

    def replace_auxiliary_attack_configs(self, target_nums: int = 15):
        plain_train_label = np.array(
            LabelEncoder().fit_transform(self.plain_train_label)
        )
        plain_test_label = np.array(LabelEncoder().fit_transform(self.plain_test_label))
        target_class = 1
        target_indexes = np.where(plain_train_label == target_class)[0]
        target_set = np.random.choice(target_indexes, target_nums, replace=False)
        train_poison_set = np.random.choice(
            range(len(plain_train_label)), 100, replace=False
        )
        plain_alice_train_data = self.plain_alice_train_data.copy()
        train_poison_data = []
        my_sparse_cols = []
        has_dense = False
        for col in plain_alice_train_data.columns:
            if col in sparse_classes.keys():
                lbe = LabelEncoder()
                v = lbe.fit_transform(plain_alice_train_data[col].fillna('0'))
                train_poison_data.append(v[train_poison_set])
                my_sparse_cols.append(col)
            else:
                has_dense = True
        if has_dense:
            df = plain_alice_train_data.drop(columns=my_sparse_cols).fillna(0)
            mms = MinMaxScaler(feature_range=(0, 1))
            v = mms.fit_transform(df).astype(np.float32)
            train_poison_data.insert(0, v[train_poison_set])
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

    def exploit_label_counts(self) -> Tuple[int, int]:
        neg, pos = np.bincount(self.plain_train_label['Label'])
        return neg, pos
