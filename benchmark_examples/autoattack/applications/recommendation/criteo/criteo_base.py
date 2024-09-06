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
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
    DatasetType,
    InputMode,
)
from benchmark_examples.autoattack.global_config import is_simple_test
from benchmark_examples.autoattack.utils.data_utils import get_sample_indexes
from benchmark_examples.autoattack.utils.resources import ResourceDict, ResourcesPack
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
        'C1': 1261,
        'C2': 531,
        'C3': 321439,
        'C4': 120965,
        'C5': 267,
        'C6': 16,
        'C7': 10863,
        'C8': 563,
        'C9': 3,
        'C10': 30792,
        'C11': 4731,
        'C12': 268488,
        'C13': 3068,
        'C14': 26,
        'C15': 8934,
        'C16': 205924,
        'C17': 10,
        'C18': 3881,
        'C19': 1855,
        'C20': 4,
        'C21': 240748,
        'C22': 16,
        'C23': 15,
        'C24': 41283,
        'C25': 70,
        'C26': 30956,
    }
)


class CriteoDataset(Dataset):
    def __init__(self, x, enable_label=0, indexes: np.ndarray = None):
        df = x[0].fillna('-1')
        self.enable_label = enable_label
        self.tensors = []
        my_sparse_cols = []
        has_dense = False
        for col in df.columns:
            if col in sparse_classes.keys():
                lbe = LabelEncoder()
                v = lbe.fit_transform(df[col].fillna('0'))
                my_sparse_cols.append(col)
                if indexes is not None:
                    v = v[indexes]
                self.tensors.append(torch.tensor(v, dtype=torch.long))
            else:
                has_dense = True
        if has_dense:
            df = df.drop(columns=my_sparse_cols)
            df = df.fillna(0)
            mms = MinMaxScaler(feature_range=(0, 1))
            self.dense_array = mms.fit_transform(df).astype(np.float32)
            if indexes is not None:
                self.dense_array = self.dense_array[indexes]
            self.tensors.insert(0, torch.tensor(self.dense_array))
        if enable_label != 1:
            self.label_tensor = torch.tensor(x[1].values.astype(np.float32))

    def __getitem__(self, index):
        if self.enable_label != 1:
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


def process_data(data: pd.DataFrame):
    data = data.fillna('-1')
    sparse_cols = []
    for col in data.columns:
        if col in sparse_classes.keys():
            sparse_cols.append(col)
            lbe = LabelEncoder()
            data[col] = lbe.fit_transform(data[col])
    data = data.drop(columns=sparse_cols)
    data = data.fillna(0)
    mms = MinMaxScaler(feature_range=(0, 1))
    data = mms.fit_transform(data).astype(np.float32)
    return data


class CriteoBase(ApplicationBase, ABC):
    def __init__(
        self,
        alice,
        bob,
        epoch=1,
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
            alice,
            bob,
            has_custom_dataset=True,
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
        self.train_dataset_len = 800000
        self.test_dataset_len = 200000
        if global_config.is_simple_test():
            self.train_dataset_len = 800
            self.test_dataset_len = 200

    def dataset_name(self):
        return 'criteo'

    def set_config(self, config: Dict[str, str] | None):
        super().set_config(config)
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

    def prepare_data(self):
        random_state = 1234
        num_samples = 1000 if is_simple_test() else 1000000
        # need to read label, so + 1
        data = datasets.load_criteo(
            {
                self.alice: (1, self.alice_fea_nums + 1),
                self.bob: (self.alice_fea_nums + 1, 40),
            },
            num_samples=num_samples,
        )
        label = datasets.load_criteo({self.bob: (0, 1)}, num_samples=num_samples)
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )
        return train_data, train_label, test_data, test_label

    def create_dataset_builder_alice(self):
        train_batch_size = self.train_batch_size

        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = CriteoDataset(x, enable_label=1)
            dataloader = torch_data.DataLoader(
                dataset=data_set, batch_size=train_batch_size
            )
            return dataloader

        return dataset_builder

    def create_dataset_builder_bob(self):
        train_batch_size = self.train_batch_size

        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = CriteoDataset(x, enable_label=0)
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

    def get_plain_train_alice_data(self):
        if self._plain_train_alice_data is not None:
            return self._plain_train_alice_data

        self._plain_train_alice_data = reveal(
            self.get_train_data().partitions[self.alice].data
        )
        return self.get_plain_train_alice_data()

    def get_plain_train_bob_data(self):
        if self._plain_train_bob_data is not None:
            return self._plain_train_bob_data

        self._plain_train_bob_data = reveal(
            self.get_train_data().partitions[self.bob].data
        )
        return self.get_plain_train_bob_data()

    def get_plain_test_alice_data(self):
        if self._plain_test_alice_data is not None:
            return self._plain_test_alice_data

        self._plain_test_alice_data = reveal(
            self.get_test_data().partitions[self.alice].data
        )
        return self.get_plain_test_alice_data()

    def get_plain_test_bob_data(self):
        if self._plain_test_bob_data is not None:
            return self._plain_test_bob_data

        self._plain_test_bob_data = reveal(
            self.get_test_data().partitions[self.bob].data
        )
        return self.get_plain_test_bob_data()

    def get_device_f_train_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 1,
        **kwargs,
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        x = [self.get_plain_train_device_f_data()]
        if enable_label == 0:
            x.append(self.get_plain_train_label())
        return CriteoDataset(
            x,
            enable_label=enable_label,
            indexes=indexes,
        )

    def get_device_y_train_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 0,
        **kwargs,
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        x = [self.get_plain_train_device_y_data()]
        if enable_label == 0:
            x.append(self.get_plain_train_label())
        return CriteoDataset(
            x,
            enable_label=enable_label,
            indexes=indexes,
        )

    def get_device_f_test_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 1,
        **kwargs,
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        x = [self.get_plain_test_device_f_data()]
        if enable_label == 0:
            x.append(self.get_plain_test_label())
        return CriteoDataset(
            x,
            enable_label=enable_label,
            indexes=indexes,
        )

    def get_device_y_test_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 0,
        **kwargs,
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        x = [self.get_plain_test_device_y_data()]
        if enable_label == 0:
            x.append(self.get_plain_test_label())
        return CriteoDataset(
            [self.get_plain_test_device_y_data(), self.get_plain_test_label()],
            enable_label=enable_label,
            indexes=indexes,
        )

    def resources_consumption(self) -> ResourcesPack:
        # 1786MiB
        return (
            ResourcesPack()
            .with_debug_resources(
                ResourceDict(
                    gpu_mem=2 * 1024 * 1024 * 1024, CPU=1, memory=3 * 1024 * 1024 * 1024
                )
            )
            .with_sim_resources(
                self.device_y.party,
                ResourceDict(
                    gpu_mem=2 * 1024 * 1024 * 1024, CPU=1, memory=3 * 1024 * 1024 * 1024
                ),
            )
            .with_sim_resources(
                self.device_f.party,
                ResourceDict(
                    gpu_mem=1.8 * 1024 * 1024 * 1024,
                    CPU=1,
                    memory=3 * 1024 * 1024 * 1024,
                ),
            )
        )

    def tune_metrics(self) -> Dict[str, str]:
        return {
            "train_BinaryAccuracy": "max",
            "train_BinaryPrecision": "max",
            "train_BinaryAUROC": "max",
            "val_BinaryAccuracy": "max",
            "val_BinaryPrecision": "max",
            "val_BinaryAUROC": "max",
        }

    def classfication_type(self) -> ClassficationType:
        return ClassficationType.BINARY

    def base_input_mode(self) -> InputMode:
        return InputMode.MULTI

    def dataset_type(self) -> DatasetType:
        return DatasetType.RECOMMENDATION
