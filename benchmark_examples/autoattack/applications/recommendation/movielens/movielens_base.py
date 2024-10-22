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

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
    DatasetType,
    InputMode,
)
from benchmark_examples.autoattack.global_config import is_simple_test
from benchmark_examples.autoattack.utils.data_utils import (
    SparseTensorDataset,
    get_sample_indexes,
)
from benchmark_examples.autoattack.utils.resources import ResourceDict, ResourcesPack
from secretflow.data.split import train_test_split
from secretflow.utils.simulation.datasets import load_ml_1m

NUM_USERS = 6040
NUM_MOVIES = 3952
GENDER_VOCAB = ["F", "M"]
AGE_VOCAB = [1, 18, 25, 35, 45, 50, 56]
OCCUPATION_VOCAB = [i for i in range(21)]
GENRES_VOCAB = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def user_preprocess(series):
    return [hash(v) % NUM_USERS for v in series]


def gender_preprocess(series):
    return [
        GENDER_VOCAB.index(word) if word in GENDER_VOCAB else len(GENDER_VOCAB)
        for word in series
    ]


def age_preprocess(series):
    return [
        AGE_VOCAB.index(word) if word in AGE_VOCAB else len(AGE_VOCAB)
        for word in series
    ]


def occupation_preprocess(series):
    return [
        (
            OCCUPATION_VOCAB.index(word)
            if word in OCCUPATION_VOCAB
            else len(OCCUPATION_VOCAB)
        )
        for word in series
    ]


def movie_preprocess(series):
    return [hash(v) % NUM_MOVIES for v in series]


def genres_preprocess(series):
    indices = []
    for sentence in series:
        words = sentence.split()
        is_in = False
        for word in words:
            if word in GENRES_VOCAB:
                indices.append(GENRES_VOCAB.index(word))
                is_in = True
                break
        if not is_in:
            indices.append(len(GENRES_VOCAB))
    return indices


all_features = OrderedDict(
    {
        "UserID": user_preprocess,
        "Gender": gender_preprocess,
        "Age": age_preprocess,
        "Occupation": occupation_preprocess,
        # split --------
        "MovieID": movie_preprocess,
        "Genres": genres_preprocess,
    }
)

feature_classes = OrderedDict(
    {
        "UserID": NUM_USERS,
        "Gender": 2,
        "Age": len(AGE_VOCAB),
        "Occupation": len(OCCUPATION_VOCAB),
        # split --------
        "MovieID": NUM_MOVIES,
        "Genres": len(GENRES_VOCAB),
    }
)


def process_data(data: pd.DataFrame):
    for col in data.columns:
        data[col] = all_features[col](data[col])
    return data


def process_label(label: pd.DataFrame):
    label['Rating'] = pd.Series([0 if int(v) < 3 else 1 for v in label['Rating']])
    return label


class MovielensBase(ApplicationBase, ABC):
    def __init__(
        self,
        alice,
        bob,
        epoch=4,
        train_batch_size=128,
        hidden_size=64,
        alice_fea_nums=4,
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
            total_fea_nums=6,
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
        self.alice_input_dims = None
        self.bob_input_dims = None
        self.train_dataset_len = 800167
        self.test_dataset_len = 200042
        if global_config.is_simple_test():
            self.train_dataset_len = 800
            self.test_dataset_len = 200

    def dataset_name(self):
        return 'movielens'

    def set_config(self, config: Dict[str, str] | None):
        super().set_config(config)
        self.alice_input_dims = [
            list(feature_classes.values())[i] for i in range(self.alice_fea_nums)
        ]
        self.bob_input_dims = [
            list(feature_classes.values())[i + self.alice_fea_nums]
            for i in range(self.bob_fea_nums)
        ]

    def prepare_data(self):
        print([list(all_features.keys())[i] for i in range(self.alice_fea_nums)])
        print(
            [
                list(all_features.keys())[i + self.alice_fea_nums]
                for i in range(self.bob_fea_nums)
            ]
            + ['Rating']
        )
        vdf = load_ml_1m(
            part={
                self.alice: [
                    list(all_features.keys())[i] for i in range(self.alice_fea_nums)
                ],
                self.bob: [
                    list(all_features.keys())[i + self.alice_fea_nums]
                    for i in range(self.bob_fea_nums)
                ]
                + ['Rating'],
            },
            num_sample=1000 if is_simple_test() else -1,
        )
        label = vdf['Rating']
        data = vdf.drop(columns=['Rating'])
        data["UserID"] = data["UserID"].astype("string")
        data["MovieID"] = data["MovieID"].astype("string")
        data = data.apply_func(process_data)
        label = label.apply_func(process_label)
        data = data.values
        label = label.values
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=global_config.get_random_seed()
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=global_config.get_random_seed()
        )

        return train_data, train_label, test_data, test_label

    def create_dataset_builder_alice(self):
        batch_size = self.train_batch_size

        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = SparseTensorDataset(x)
            dataloader = torch_data.DataLoader(
                dataset=data_set,
                batch_size=batch_size,
            )
            return dataloader

        return dataset_builder

    def create_dataset_builder_bob(self):
        batch_size = self.train_batch_size

        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = SparseTensorDataset(x)
            dataloader = torch_data.DataLoader(
                dataset=data_set,
                batch_size=batch_size,
            )
            return dataloader

        return dataset_builder

    def create_predict_dataset_builder_alice(
        self, *args, **kwargs
    ) -> Optional[Callable]:
        return self.create_dataset_builder_alice()

    def create_predict_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return self.create_dataset_builder_alice()

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
        return SparseTensorDataset(
            x,
            indexes=indexes,
            enable_label=enable_label,
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
        return SparseTensorDataset(
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
        return SparseTensorDataset(
            x,
            indexes=indexes,
            enable_label=enable_label,
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
        return SparseTensorDataset(
            x,
            enable_label=enable_label,
            indexes=indexes,
        )

    def resources_consumption(self) -> ResourcesPack:
        # 750MiB
        return (
            ResourcesPack()
            .with_debug_resources(
                ResourceDict(
                    gpu_mem=2 * 1024 * 1024 * 1024,
                    CPU=1,
                    memory=1.6 * 1024 * 1024 * 1024,
                )
            )
            .with_sim_resources(
                self.device_y.party,
                ResourceDict(
                    gpu_mem=2 * 1024 * 1024 * 1024,
                    CPU=1,
                    memory=1.6 * 1024 * 1024 * 1024,
                ),
            )
            .with_sim_resources(
                self.device_f.party,
                ResourceDict(
                    gpu_mem=1.5 * 1024 * 1024 * 1024,
                    CPU=1,
                    memory=1.6 * 1024 * 1024 * 1024,
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
