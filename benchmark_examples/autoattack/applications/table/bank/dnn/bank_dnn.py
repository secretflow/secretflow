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
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow import reveal
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper
from secretflow.preprocessing import LabelEncoder, MinMaxScaler
from secretflow.utils.simulation.datasets import load_bank_marketing


class BankDnn(TrainBase):
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
        self.hidden_size = config.get('hidden_size', hidden_size)
        self.alice_fea_nums = config.get('alice_fea_nums', alice_fea_nums)
        self.bob_fea_nums = 16 - self.alice_fea_nums
        self.dnn_units_size = [100, self.hidden_size]
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
        history = sl_model.fit(
            self.train_data,
            self.train_label,
            validation_data=(self.test_data, self.test_label),
            epochs=self.epoch,
            batch_size=self.train_batch_size,
            shuffle=False,
            verbose=1,
            validation_freq=1,
            callbacks=callbacks,
        )
        logging.warning(history['val_BinaryAccuracy'])

    def _prepare_data(self):
        data = load_bank_marketing(
            parts={
                self.alice: (0, self.alice_fea_nums),
                self.bob: (self.alice_fea_nums, 16),
            },
            axis=1,
        )
        label = load_bank_marketing(parts={self.alice: (16, 17)}, axis=1)

        encoder = LabelEncoder()
        data['job'] = encoder.fit_transform(data['job'])
        data['marital'] = encoder.fit_transform(data['marital'])
        data['education'] = encoder.fit_transform(data['education'])
        data['default'] = encoder.fit_transform(data['default'])
        data['housing'] = encoder.fit_transform(data['housing'])
        data['loan'] = encoder.fit_transform(data['loan'])
        data['contact'] = encoder.fit_transform(data['contact'])
        data['poutcome'] = encoder.fit_transform(data['poutcome'])
        data['month'] = encoder.fit_transform(data['month'])
        label = encoder.fit_transform(label).astype(np.float32)
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data).astype(np.float32)  # 因为模型是32的
        random_state = 1234
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )

        return train_data, train_label, test_data, test_label

    def create_base_model(self, input_dim):
        model = TorchModel(
            model_fn=DnnBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(Precision, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[input_dim],
            dnn_units_size=self.dnn_units_size,
        )
        return model

    def _create_base_model_alice(self):
        return self.create_base_model(self.alice_fea_nums)

    def _create_base_model_bob(self):
        return self.create_base_model(self.bob_fea_nums)

    def _create_fuse_model(self):
        return TorchModel(
            model_fn=DnnFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(Precision, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=[1],
            output_func=nn.Sigmoid,
        )

    def support_attacks(self):
        return ['norm', 'lia']

    def lia_auxiliary_data_builder(self, batch_size=16, file_path=None):
        def split_some_data(df, label):
            df['y'] = label['y']
            df = df.sample(frac=1, random_state=42)[: int(len(df) * 0.2)]
            label = df['y']
            df = df.drop(columns=['y'])
            logging.warning(f"label shape = {torch.LongTensor(label.values).shape}")
            datasets = TensorDataset(
                torch.tensor(df.values), torch.LongTensor(label.values)
            )
            return DataLoader(
                datasets,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )

        def prepare_data():
            train = reveal(self.train_data.partitions[self.bob].data)
            tr_label = reveal(self.train_label.partitions[self.alice].data)
            test = reveal(self.test_data.partitions[self.bob].data)
            tst_label = reveal(self.test_label.partitions[self.alice].data)
            logging.warning(f"test label = {tst_label}")
            train_complete_trainloader = DataLoader(
                TensorDataset(
                    torch.tensor(train.values), torch.LongTensor(tr_label['y'].values)
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            train_labeled_dataloader = split_some_data(train, tr_label)
            tr_label['y'] = -1
            train_unlabeled_dataloader = split_some_data(train, tr_label)
            test_loader = DataLoader(
                TensorDataset(
                    torch.tensor(test.values), torch.LongTensor(tst_label['y'].values)
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            return (
                train_labeled_dataloader,
                train_unlabeled_dataloader,
                test_loader,
                train_complete_trainloader,
            )

        return prepare_data

    def lia_auxiliary_model(self, ema=False):
        from benchmark_examples.autoattack.attacks.lia import BottomModelPlus

        bottom_model = DnnBase(
            input_dims=[self.alice_fea_nums], dnn_units_size=self.dnn_units_size
        )
        model = BottomModelPlus(
            bottom_model, size_bottom_out=self.dnn_units_size[-1], num_classes=2
        )

        if ema:
            for param in model.parameters():
                param.detach_()

        return model
