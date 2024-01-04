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
import torch.nn as nn
import torch.optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchmetrics import AUROC, Accuracy

from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.applications.sl_deepfm_torch import DeepFMBase, DeepFMFuse
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper
from secretflow.utils.simulation.datasets import load_bank_marketing

all_features = OrderedDict(
    {
        'age': 100,
        'job': 12,
        'marital': 3,
        'education': 4,
        # default split ----
        'default': 2,
        'balance': 2353,
        'housing': 2,
        'loan': 2,
        'contact': 3,
        'day': 31,
        'month': 12,
        'duration': 875,
        'campaign': 32,
        'pdays': 292,
        'previous': 24,
        'poutcome': 4,
    }
)


class AliceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label):
        self.tensors = []
        for feat in df.columns:
            v = df[feat]
            v = v.reset_index(drop=True)
            lbe = LabelEncoder()
            v = lbe.fit_transform(v)
            self.tensors.append(torch.tensor(v))
        lbe = LabelEncoder()
        self.label = torch.unsqueeze(
            torch.tensor(lbe.fit_transform(label).astype(np.float32)), dim=1
        )

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), self.label[index]

    def __len__(self):
        return self.tensors[0].size(0)


class BobDataset(Dataset):
    def __init__(self, df):
        self.tensors = []
        for feas in df.columns:
            v = df[feas].reset_index(drop=True)
            lbe = LabelEncoder()
            v = lbe.fit_transform(v)
            self.tensors.append(torch.tensor(v))

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class BankDeepfm(TrainBase):
    def __init__(
        self,
        config,
        alice,
        bob,
        epoch=10,
        train_batch_size=128,
        hidden_size=64,
        alice_fea_nums=9,
    ):
        self.hidden_size = hidden_size
        self.alice_fea_nums = config.get('alice_fea_nums', alice_fea_nums)
        self.bob_fea_nums = 16 - self.alice_fea_nums
        self.alice_fea_classes = {
            list(all_features.keys())[i]: all_features[list(all_features.keys())[i]]
            for i in range(self.alice_fea_nums)
        }
        self.bob_fea_classes = {
            list(all_features.keys())[i + self.alice_fea_nums]: all_features[
                list(all_features.keys())[i + self.alice_fea_nums]
            ]
            for i in range(self.bob_fea_nums)
        }
        super().__init__(
            config, alice, bob, alice, 2, epoch=epoch, train_batch_size=train_batch_size
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

    def _prepare_data(self):
        data = load_bank_marketing(
            parts={
                self.alice: (0, self.alice_fea_nums),
                self.bob: (self.alice_fea_nums, 16),
            },
            axis=1,
        )
        label = load_bank_marketing(parts={self.alice: (16, 17)}, axis=1)
        random_state = 1234
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )
        return train_data, train_label, test_data, test_label

    def _create_base_model_alice(self):
        model = TorchModel(
            model_fn=DeepFMBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[v for v in self.alice_fea_classes.values()],
            dnn_units_size=[100, self.hidden_size],
        )
        return model

    def _create_base_model_bob(self):
        model = TorchModel(
            model_fn=DeepFMBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[v for v in self.bob_fea_classes.values()],
            dnn_units_size=[100, self.hidden_size],
        )
        return model

    def _create_fuse_model(self):
        return TorchModel(
            model_fn=DeepFMFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=[64],
        )

    @staticmethod
    def create_dataset_builder_alice(
        batch_size=128,
        repeat_count=5,
    ):
        def dataset_builder(x):
            import torch.utils.data as torch_data

            data_set = AliceDataset(x[0], x[1])
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

            data_set = BobDataset(x[0])
            dataloader = torch_data.DataLoader(
                dataset=data_set,
                batch_size=batch_size,
            )
            return dataloader

        return dataset_builder

    def support_attacks(self):
        return ['norm']
