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

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset
from torchmetrics import AUROC, Accuracy

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import InputMode, ModelType
from benchmark_examples.autoattack.applications.table.bank.bank_base import BankBase
from benchmark_examples.autoattack.utils.data_utils import (
    SparseTensorDataset,
    create_custom_dataset_builder,
    get_sample_indexes,
)
from secretflow.data import FedNdarray
from secretflow.data.split import train_test_split
from secretflow.ml.nn.applications.sl_deepfm_torch import DeepFMBase, DeepFMFuse
from secretflow.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper


class AliceDataset(Dataset):
    def __init__(self, x):
        df = x[0]
        label = x[1]
        self.tensors = []
        for feat in range(df.shape[1]):
            self.tensors.append(torch.tensor(df[:, feat], dtype=torch.int64))
        self.label = torch.tensor(label.astype(np.float32))

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), self.label[index]

    def __len__(self):
        return self.tensors[0].size(0)


class BobDataset(Dataset):
    def __init__(self, x):
        df = x[0]
        self.tensors = []
        for feas in range(df.shape[1]):
            self.tensors.append(torch.tensor(df[:, feas], dtype=torch.int64))

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class BankDeepfm(BankBase):
    def __init__(self, alice, bob, hidden_size=64):
        super().__init__(
            alice,
            bob,
            has_custom_dataset=True,
            hidden_size=32,
            dnn_base_units_size_alice=[128, -1],
            dnn_base_units_size_bob=[128, -1],
            dnn_fuse_units_size=[64, 64, 1],
            deepfm_embedding_dim=8,
        )
        self.metrics = [
            metric_wrapper(Accuracy, task="binary"),
            metric_wrapper(AUROC, task="binary"),
        ]

    def model_type(self) -> ModelType:
        return ModelType.DEEPFM

    def prepare_data(
        self, **kwargs
    ) -> Tuple[FedNdarray, FedNdarray, FedNdarray, FedNdarray]:
        data, label = super().load_bank_data()
        data = data.values
        label = label.values
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=global_config.get_random_seed()
        )
        train_label, test_label = train_test_split(
            label, train_size=0.8, random_state=global_config.get_random_seed()
        )
        return train_data, train_label, test_data, test_label

    def create_base_model_alice(self):
        model = TorchModel(
            model_fn=DeepFMBase,
            optim_fn=optim_wrapper(torch.optim.Adam),
            input_dims=[v for v in self.alice_fea_classes.values()],
            dnn_units_size=self.dnn_base_units_size_alice,
            fm_embedding_dim=self.deepfm_embedding_dim,
        )
        return model

    def create_base_model_bob(self):
        model = TorchModel(
            model_fn=DeepFMBase,
            optim_fn=optim_wrapper(torch.optim.Adam),
            input_dims=[v for v in self.bob_fea_classes.values()],
            dnn_units_size=self.dnn_base_units_size_bob,
            fm_embedding_dim=self.deepfm_embedding_dim,
        )
        return model

    def create_fuse_model(self):
        return TorchModel(
            model_fn=DeepFMFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=self.metrics,
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=self.dnn_fuse_units_size,
            output_func=nn.Sigmoid,
        )

    def create_dataset_builder_alice(self):
        batch_size = self.train_batch_size
        return create_custom_dataset_builder(SparseTensorDataset, batch_size)

    def create_dataset_builder_bob(self):
        batch_size = self.train_batch_size
        return create_custom_dataset_builder(SparseTensorDataset, batch_size)

    def create_predict_dataset_builder_alice(
        self, *args, **kwargs
    ) -> Optional[Callable]:
        return create_custom_dataset_builder(SparseTensorDataset, self.train_batch_size)

    def create_predict_dataset_builder_bob(self, *args, **kwargs) -> Optional[Callable]:
        return create_custom_dataset_builder(SparseTensorDataset, self.train_batch_size)

    def get_device_f_train_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 1,
        **kwargs
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        x = [self.get_plain_train_device_f_data()]
        if enable_label == 0:
            x.append(self.get_plain_train_label())
        return SparseTensorDataset(x, indexes=indexes, enable_label=enable_label)

    def get_device_y_train_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 0,
        **kwargs
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        x = [self.get_plain_train_device_y_data()]
        if enable_label == 0:
            x.append(self.get_plain_train_label())
        return SparseTensorDataset(
            x,
            indexes=indexes,
        )

    def get_device_f_test_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 1,
        **kwargs
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        x = [self.get_plain_test_device_f_data()]
        if enable_label == 0:
            x.append(self.get_plain_test_label())
        return SparseTensorDataset(x, indexes=indexes, enable_label=enable_label)

    def get_device_y_test_dataset(
        self,
        sample_size: int = None,
        frac: float = None,
        indexes: np.ndarray = None,
        enable_label: int = 0,
        **kwargs
    ):
        indexes = get_sample_indexes(self.train_dataset_len, sample_size, frac, indexes)
        x = [self.get_plain_test_device_y_data()]
        if enable_label == 0:
            x.append(self.get_plain_test_label())
        return SparseTensorDataset(
            x,
            indexes=indexes,
            enable_label=enable_label,
        )

    def tune_metrics(self) -> Dict[str, str]:
        return {
            "train_BinaryAccuracy": "max",
            "train_BinaryAUROC": "max",
            "val_BinaryAccuracy": "max",
            "val_BinaryAUROC": "max",
        }

    def base_input_mode(self) -> InputMode:
        return InputMode.MULTI
