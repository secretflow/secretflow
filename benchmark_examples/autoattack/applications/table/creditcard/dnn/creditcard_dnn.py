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

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
    DatasetType,
    InputMode,
    ModelType,
)
from benchmark_examples.autoattack.utils.resources import ResourceDict, ResourcesPack
from secretflow.data import FedNdarray
from secretflow.data.split import train_test_split
from secretflow.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow.preprocessing import StandardScaler
from secretflow.utils.simulation.datasets import load_creditcard


class CreditcardDnn(ApplicationBase):

    def __init__(self, alice, bob):
        super().__init__(
            alice,
            bob,
            device_y=bob,
            total_fea_nums=29,
            alice_fea_nums=25,
            num_classes=2,
            epoch=2,
            train_batch_size=1024,
            hidden_size=28,
            dnn_base_units_size_alice=[int(28 / 2), -1],
            dnn_base_units_size_bob=[4, -1],
            dnn_fuse_units_size=[1],
        )

    def dataset_name(self):
        return 'creditcard'

    def prepare_data(self) -> Tuple[FedNdarray, FedNdarray, FedNdarray, FedNdarray]:
        num_sample = 2841 if global_config.is_simple_test() else 284160
        data = load_creditcard(
            {self.alice: (0, 25), self.bob: (25, 29)}, num_sample=num_sample
        )
        label = load_creditcard({self.bob: (29, 30)}, num_sample=num_sample).astype(
            np.float32
        )
        scaler = StandardScaler()
        data = scaler.fit_transform(data).astype(np.float32)
        random_state = 1234
        data = data.values
        label = label.values

        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=random_state
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=random_state
        )
        return train_data, train_label, test_data, test_label

    def model_type(self) -> ModelType:
        return ModelType.DNN

    def create_base_model_alice(self):
        return TorchModel(
            model_fn=DnnBase,
            optim_fn=optim_wrapper(torch.optim.Adam),
            input_dims=[25],
            dnn_units_size=self.dnn_base_units_size_alice,
        )

    def create_base_model_bob(self):
        return TorchModel(
            model_fn=DnnBase,
            optim_fn=optim_wrapper(torch.optim.Adam),
            input_dims=[4],
            dnn_units_size=self.dnn_base_units_size_bob,
        )

    def create_fuse_model(self):
        return TorchModel(
            model_fn=DnnFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=[1],
            output_func=nn.Sigmoid,
        )

    def tune_metrics(self) -> Dict[str, str]:
        return {
            "train_BinaryAccuracy": "max",
            "val_BinaryAccuracy": "max",
        }

    def classfication_type(self) -> ClassficationType:
        return ClassficationType.BINARY

    def base_input_mode(self) -> InputMode:
        return InputMode.SINGLE

    def dataset_type(self) -> DatasetType:
        return DatasetType.TABLE

    def resources_consumption(self) -> ResourcesPack:
        # 582MiB
        return (
            ResourcesPack()
            .with_debug_resources(
                ResourceDict(
                    gpu_mem=600 * 1024 * 1024, CPU=1, memory=1.5 * 1024 * 1024 * 1024
                )
            )
            .with_sim_resources(
                self.device_y.party,
                ResourceDict(
                    gpu_mem=600 * 1024 * 1024, CPU=1, memory=1.5 * 1024 * 1024 * 1024
                ),
            )
            .with_sim_resources(
                self.device_f.party,
                ResourceDict(
                    gpu_mem=500 * 1024 * 1024, CPU=1, memory=1.5 * 1024 * 1024 * 1024
                ),
            )
        )
