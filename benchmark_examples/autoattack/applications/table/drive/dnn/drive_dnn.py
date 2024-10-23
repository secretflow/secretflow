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

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
    DatasetType,
    InputMode,
    ModelType,
)
from benchmark_examples.autoattack.utils.data_utils import sample_ndarray
from benchmark_examples.autoattack.utils.resources import ResourceDict, ResourcesPack
from secretflow.data.ndarray import FedNdarray, PartitionWay
from secretflow.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper


class DriveDnn(ApplicationBase):
    def __init__(self, alice, bob):
        super().__init__(
            alice,
            bob,
            device_y=alice,
            total_fea_nums=48,
            alice_fea_nums=28,
            num_classes=11,
            epoch=2,
            train_batch_size=64,
            hidden_size=64,
            dnn_base_units_size_alice=[-1],
            dnn_base_units_size_bob=[-1],
            dnn_fuse_units_size=[600, 300, 100, 11],
        )
        self.metrics = [
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=11, average='micro'
            ),
            metric_wrapper(
                Precision, task="multiclass", num_classes=11, average='micro'
            ),
            metric_wrapper(AUROC, task="multiclass", num_classes=11),
        ]

    def dataset_name(self):
        return 'drive'

    def prepare_data(
        self,
    ):
        from secretflow.utils.simulation.datasets import _DATASETS, get_dataset

        path = get_dataset(_DATASETS['drive_cleaned'])
        full_data_table = np.genfromtxt(path, delimiter=',')
        if global_config.is_simple_test():
            full_data_table, _ = sample_ndarray(full_data_table, sample_size=2000)
        samples = full_data_table[:, :-1].astype(np.float32)
        # permuate columns
        batch, columns = samples.shape
        permu_cols = torch.randperm(columns)
        samples = samples[:, permu_cols]

        labels = full_data_table[:, -1].astype(np.int64)
        fea_min = samples.min(axis=0)
        fea_max = samples.max(axis=0)

        samples = (samples - fea_min) / (fea_max - fea_min)
        mean_attr = samples.mean(axis=0)
        var_attr = samples.var(axis=0)

        random_selection = np.random.rand(samples.shape[0]) <= 0.6
        train_sample = samples[random_selection]
        train_label = labels[random_selection]
        sample_left = samples[~random_selection]
        label_left = labels[~random_selection]

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
        train_label = FedNdarray(
            partitions={self.device_y: self.device_y(lambda x: x)(train_label)},
            partition_way=PartitionWay.VERTICAL,
        )
        test_label = FedNdarray(
            partitions={self.device_y: self.device_y(lambda x: x)(test_label)},
            partition_way=PartitionWay.VERTICAL,
        )
        # self.mean_attr = mean_attr
        return train_data, train_label, test_data, test_label

    def model_type(self) -> ModelType:
        return ModelType.DNN

    def create_base_model_alice(self):
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(torch.optim.Adam)
        return TorchModel(
            model_fn=DnnBase,
            optim_fn=optim_fn,
            input_dims=[28],
            dnn_units_size=self.dnn_base_units_size_alice,
        )

    def create_base_model_bob(self):
        optim_fn = optim_wrapper(torch.optim.Adam)
        return TorchModel(
            model_fn=DnnBase,
            optim_fn=optim_fn,
            input_dims=[20],
            dnn_units_size=self.dnn_base_units_size_bob,
        )

    def create_fuse_model(self):
        loss_fn = nn.CrossEntropyLoss
        optim_fn = optim_wrapper(torch.optim.Adam)
        return TorchModel(
            model_fn=DnnFuse,
            loss_fn=loss_fn,
            optim_fn=optim_fn,
            metrics=self.metrics,
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=self.dnn_fuse_units_size,
            output_func=None,
        )

    def tune_metrics(self) -> Dict[str, str]:
        return {
            "train_MulticlassAccuracy": "max",
            "train_MulticlassPrecision": "max",
            "train_MulticlassAUROC": "max",
            "val_MulticlassAccuracy": "max",
            "val_MulticlassPrecision": "max",
            "val_MulticlassAUROC": "max",
        }

    def classfication_type(self) -> ClassficationType:
        return ClassficationType.MULTICLASS

    def base_input_mode(self) -> InputMode:
        return InputMode.SINGLE

    def dataset_type(self) -> DatasetType:
        return DatasetType.TABLE

    def resources_consumption(self) -> ResourcesPack:
        # 480MB
        return (
            ResourcesPack()
            .with_debug_resources(
                ResourceDict(gpu_mem=500 * 1024 * 1024, CPU=1, memory=800 * 1024 * 1024)
            )
            .with_sim_resources(
                self.device_y.party,
                ResourceDict(
                    gpu_mem=500 * 1024 * 1024, CPU=1, memory=800 * 1024 * 1024
                ),
            )
            .with_sim_resources(
                self.device_f.party,
                ResourceDict(
                    gpu_mem=400 * 1024 * 1024, CPU=1, memory=800 * 1024 * 1024
                ),
            )
        )
