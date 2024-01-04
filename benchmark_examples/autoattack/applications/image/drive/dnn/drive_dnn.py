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
from torchmetrics import Accuracy, Precision

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.fl.utils import metric_wrapper, optim_wrapper
from secretflow.ml.nn.utils import BaseModule, TorchModel


class SLBaseNet(BaseModule):
    def __init__(self):
        super(SLBaseNet, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        y = x
        return y

    def output_num(self):
        return 1


class SLFuseModel(BaseModule):
    def __init__(self, input_dim=48, output_dim=11):
        super(SLFuseModel, self).__init__()
        torch.manual_seed(1234)
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim),
        )

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return self.dense(x)


class DriveDnn(TrainBase):
    def __init__(self, config, alice, bob):
        self.hidden_size = 64
        super().__init__(config, alice, bob, alice, 2, train_batch_size=64)

    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        base_model_dict = {
            self.alice: self.alice_base_model,
            self.bob: self.bob_base_model,
        }
        sl_model = SLModel(
            base_model_dict=base_model_dict,
            device_y=self.device_y,
            model_fuse=self.fuse_model,
            dp_strategy_dict=None,
            compressor=None,
            simulation=True,
            random_seed=1234,
            backend='torch',
            strategy='split_nn',
        )
        sl_model.fit(
            self.train_data,
            self.train_label,
            validation_data=(self.test_data, self.test_label),
            epochs=3,
            batch_size=self.train_batch_size,
            shuffle=False,
            random_seed=1234,
            dataset_builder=None,
            callbacks=callbacks,
        )

    def _prepare_data(
        self,
    ):
        full_data_table = np.genfromtxt(
            global_config.get_dataset_path() + "/drive/drive_cleaned.csv",
            delimiter=',',
        )
        samples = full_data_table[:, :-1].astype(np.float32)
        # permuate columns
        batch, columns = samples.shape
        print(batch, columns)
        permu_cols = torch.randperm(columns)
        logging.info("Dataset column permutation is: \n %s", permu_cols)
        samples = samples[:, permu_cols]

        labels = full_data_table[:, -1].astype(np.long)
        fea_min = samples.min(axis=0)
        fea_max = samples.max(axis=0)
        print(fea_min.shape)
        print(fea_max.shape)

        samples = (samples - fea_min) / (fea_max - fea_min)
        mean_attr = samples.mean(axis=0)
        var_attr = samples.var(axis=0)
        print(mean_attr.shape, var_attr.shape)

        random_selection = np.random.rand(samples.shape[0]) <= 0.6
        print(random_selection)
        train_sample = samples[random_selection]
        train_label = labels[random_selection]
        sample_left = samples[~random_selection]
        label_left = labels[~random_selection]
        print(train_sample.shape, sample_left.shape)
        print(train_sample)

        random_selection = np.random.rand(sample_left.shape[0]) <= 0.5
        test_sample = sample_left[random_selection]
        test_label = label_left[random_selection]
        self.pred_fea = sample_left[~random_selection]
        self.pred_label = label_left[~random_selection]
        train_data = FedNdarray(
            partitions={
                self.alice: self.alice(lambda x: x[:, :28])(train_sample),
                self.bob: self.bob(lambda x: x[:, 28:])(train_sample),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        test_data = FedNdarray(
            partitions={
                self.alice: self.alice(lambda x: x[:, :28])(test_sample),
                self.bob: self.bob(lambda x: x[:, 28:])(test_sample),
            },
            partition_way=PartitionWay.VERTICAL,
        )
        train_label = self.device_y(lambda x: x)(train_label)
        test_label = self.device_y(lambda x: x)(test_label)
        self.mean_attr = mean_attr
        return (train_data, train_label, test_data, test_label)

    def create_base_model(self, input_dim, output_dim):
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(torch.optim.Adam)
        return TorchModel(
            model_fn=SLBaseNet,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=11, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=11, average='micro'
                ),
            ],
        )

    def _create_base_model_alice(self):
        return self.create_base_model(28, self.hidden_size)

    def _create_base_model_bob(self):
        return self.create_base_model(20, self.hidden_size)

    def _create_fuse_model(self):
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(torch.optim.Adam)
        return TorchModel(
            model_fn=SLFuseModel,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=11, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=11, average='micro'
                ),
            ],
        )

    def support_attacks(self):
        return ['fia']

    def fia_auxiliary_data_builder(self):
        def prepare_data():
            print('prepare_data num: ', self.pred_fea.shape)
            alice_data = self.pred_fea[:, :28]
            bob_data = self.pred_fea[:, 28:]

            alice_dataset = TensorDataset(torch.tensor(alice_data))
            alice_dataloader = DataLoader(
                dataset=alice_dataset,
                shuffle=False,
                batch_size=self.train_batch_size,
            )

            bob_dataset = TensorDataset(torch.tensor(bob_data))
            bob_dataloader = DataLoader(
                dataset=bob_dataset,
                shuffle=False,
                batch_size=self.train_batch_size,
            )

            dataloader_dict = {'alice': alice_dataloader, 'bob': bob_dataloader}
            return dataloader_dict, dataloader_dict

        return prepare_data
