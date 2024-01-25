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
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch.optim
from torch.utils.data import Dataset

from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow.data.split import train_test_split
from secretflow.data.vertical import VDataFrame
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.callbacks.callback import Callback
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
        OCCUPATION_VOCAB.index(word)
        if word in OCCUPATION_VOCAB
        else len(OCCUPATION_VOCAB)
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


class AliceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.tensors = []
        for col in df.columns:
            self.tensors.append(torch.tensor(all_features[col](df[col])))

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class BobDataset(Dataset):
    def __init__(self, df, label):
        self.tensors = []
        for col in df.columns:
            self.tensors.append(torch.tensor(all_features[col](df[col])))
        self.label = torch.unsqueeze(
            torch.tensor([0 if int(v) < 3 else 1 for v in label['Rating']]).float(),
            dim=1,
        )

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), self.label[index]

    def __len__(self):
        return self.tensors[0].size(0)


class MovielensBase(TrainBase):
    def __init__(
        self,
        config,
        alice,
        bob,
        epoch=10,
        train_batch_size=128,
        hidden_size=64,
        alice_fea_nums=4,
    ):
        self.hidden_size = hidden_size
        self.embedding_dim = 16
        self.alice_fea_nums = config.get("alice_fea_nums", alice_fea_nums)
        self.bob_fea_nums = 6 - alice_fea_nums
        self.alice_input_dims = [
            list(feature_classes.values())[i] for i in range(self.alice_fea_nums)
        ]
        self.bob_input_dims = [
            list(feature_classes.values())[i + self.alice_fea_nums]
            for i in range(self.bob_fea_nums)
        ]
        print(f"alice inpu dims = {self.alice_input_dims}")
        super().__init__(
            config, alice, bob, bob, 10, epoch=epoch, train_batch_size=train_batch_size
        )

    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        base_model_dict = {
            self.alice: self.alice_base_model,
            self.bob: self.bob_base_model,
        }
        sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=self.device_y,
            model_fuse=self.fuse_model,
            backend='torch',
        )
        data_builder_dict = {
            self.alice: self.create_dataset_builder_alice(
                batch_size=self.train_batch_size,
                repeat_count=5,
            ),
            self.bob: self.create_dataset_builder_bob(
                batch_size=self.train_batch_size,
                repeat_count=5,
            ),
        }
        history = sl_model.fit(
            self.train_data,
            self.train_label,
            validation_data=(self.test_data, self.test_label),
            epochs=self.epoch,
            batch_size=self.train_batch_size,
            shuffle=False,
            verbose=1,
            validation_freq=1,
            dataset_builder=data_builder_dict,
            callbacks=callbacks,
        )
        logging.warning(history)

    def _prepare_data(self) -> Tuple[VDataFrame, VDataFrame, VDataFrame, VDataFrame]:
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
            num_sample=10000,
        )
        label = vdf['Rating']
        data = vdf.drop(columns=['Rating'])
        # data = vdf.drop(columns=["Rating", "Timestamp", "Title", "Zip-code"])
        data["UserID"] = data["UserID"].astype("string")
        data["MovieID"] = data["MovieID"].astype("string")
        random_state = 1234
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )
        return train_data, train_label, test_data, test_label

    @staticmethod
    def create_dataset_builder_alice(
        batch_size=128,
        repeat_count=5,
    ):
        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = AliceDataset(x[0])
            dataloader = torch_data.DataLoader(
                dataset=data_set,
                batch_size=batch_size,
            )
            return dataloader

        return dataset_builder

    @staticmethod
    def create_dataset_builder_bob(
        batch_size=128,
        repeat_count=5,
    ):
        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = BobDataset(x[0], x[1])
            dataloader = torch_data.DataLoader(
                dataset=data_set,
                batch_size=batch_size,
            )
            return dataloader

        return dataset_builder
