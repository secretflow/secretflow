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
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import InputMode, ModelType
from benchmark_examples.autoattack.applications.table.bank.bank_base import BankBase
from secretflow.data import FedNdarray
from secretflow.data.split import train_test_split
from secretflow.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from secretflow.preprocessing import MinMaxScaler


class BankDnn(BankBase):
    def __init__(
        self,
        alice,
        bob,
        hidden_size=64,
    ):
        super().__init__(
            alice,
            bob,
            hidden_size=hidden_size,
            dnn_base_units_size_alice=[100, hidden_size],
            dnn_base_units_size_bob=[100, hidden_size],
            dnn_fuse_units_size=[1],
        )
        self.metrics = [
            metric_wrapper(Accuracy, task="binary"),
            metric_wrapper(Precision, task="binary"),
            metric_wrapper(AUROC, task="binary"),
        ]

    def prepare_data(self) -> Tuple[FedNdarray, FedNdarray, FedNdarray, FedNdarray]:
        data, label = super().load_bank_data()
        # binary class need float label.
        label = label.astype(np.float32)
        # after minmax dnn shows better results.
        for col in data.columns:
            mms = MinMaxScaler()
            data[col] = mms.fit_transform(data[col])
        data = data.values
        label = label.values
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=global_config.get_random_seed()
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=global_config.get_random_seed()
        )
        return train_data, train_label, test_data, test_label

    def model_type(self) -> ModelType:
        return ModelType.DNN

    def create_base_model(self, input_dim, dnn_units_size):
        model = TorchModel(
            model_fn=DnnBase,
            optim_fn=optim_wrapper(torch.optim.Adam),
            input_dims=[input_dim],
            dnn_units_size=dnn_units_size,
        )
        return model

    def create_base_model_alice(self):
        return self.create_base_model(
            self.alice_fea_nums, self.dnn_base_units_size_alice
        )

    def create_base_model_bob(self):
        return self.create_base_model(self.bob_fea_nums, self.dnn_base_units_size_bob)

    def create_fuse_model(self):
        return TorchModel(
            model_fn=DnnFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=self.metrics,
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=self.dnn_fuse_units_size,
            output_func=nn.Sigmoid,
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

    def base_input_mode(self) -> InputMode:
        return InputMode.SINGLE
