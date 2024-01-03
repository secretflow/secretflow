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
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset

from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.callbacks.callback import Callback
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
    def __init__(self, df: pd.DataFrame, label):
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
            dense_array = mms.fit_transform(df).astype(np.float32)
            self.tensors.insert(0, torch.tensor(dense_array))

        self.label_tensor = torch.tensor(label.values.astype(np.float32))

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), self.label_tensor[index]

    def __len__(self):
        return self.tensors[0].size(0)


class BobDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        df = df.fillna('-1')
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


class CriteoBase(TrainBase):
    def __init__(
        self,
        config,
        alice,
        bob,
        epoch=1,
        train_batch_size=64,
        hidden_size=64,
        alice_fea_nums=13,
    ):
        self.hidden_size = hidden_size
        # not include label
        self.alice_fea_nums = config.get('alice_fea_nums', alice_fea_nums)
        # 39 + 1 (label)
        self.bob_fea_nums = 39 - self.alice_fea_nums
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
        print("self.alice_input_dims")
        print(self.alice_input_dims)
        print("self.alice_sparse_indexes")
        print(self.alice_sparse_indexes)
        print("self.alice_dense_indexes")
        print(self.alice_dense_indexes)
        print(f"self.bob_input_dims len = {len(self.bob_input_dims)}")
        print(self.bob_input_dims)
        print(f"self.bob_sparse_indexesm len = {len(self.bob_sparse_indexes)}")
        print(self.bob_sparse_indexes)
        print("self.bob_dense_indexes")
        print(self.bob_dense_indexes)
        super().__init__(
            config, alice, bob, alice, 2, epoch=epoch, train_batch_size=train_batch_size
        )

    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        base_model_dict = {
            self.alice: self.alice_base_model,
            self.bob: self.bob_base_model,
        }
        dataset_builder_dict = {
            self.alice: self.create_dataset_builder_alice(self.train_batch_size),
            self.bob: self.create_dataset_builder_bob(self.train_batch_size),
        }
        sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=self.device_y,
            model_fuse=self.fuse_model,
            backend='torch',
        )
        history = sl_model.fit(
            self.train_data,
            self.train_label,
            validation_data=(self.test_data, self.test_label),
            epochs=self.epoch,
            batch_size=self.train_batch_size,
            shuffle=False,
            verbose=1,
            validation_freq=1,
            dataset_builder=dataset_builder_dict,
            callbacks=callbacks,
        )
        logging.warning(history['val_BinaryAccuracy'])

    def _prepare_data(self):
        random_state = 1234
        # need to read label, so + 1
        vdf = datasets.load_criteo(
            {
                self.alice: (0, self.alice_fea_nums + 1),
                self.bob: (self.alice_fea_nums + 1, 40),
            }
        )
        label = vdf['Label']
        data = vdf.drop(columns='Label', inplace=False)
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )
        return train_data, train_label, test_data, test_label

    @staticmethod
    def create_dataset_builder_alice(train_batch_size):
        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = AliceDataset(x[0], x[1])
            dataloader = torch_data.DataLoader(
                dataset=data_set, batch_size=train_batch_size
            )
            return dataloader

        return dataset_builder

    @staticmethod
    def create_dataset_builder_bob(train_batch_size):
        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = BobDataset(x[0])
            dataloader = torch_data.DataLoader(
                dataset=data_set, batch_size=train_batch_size
            )
            return dataloader

        return dataset_builder

    def support_attacks(self):
        return ['norm']
