# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applsicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack.applications.recommendation.criteo.criteo_base import (
    CriteoBase,
)
from secretflow.ml.nn.applications.sl_deepfm_torch import DeepFMBase, DeepFMFuse
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper


class CriteoDeepfm(CriteoBase):
    def __init__(self, config, alice, bob, hidden_size=64):
        super().__init__(
            config,
            alice,
            bob,
            epoch=1,
            train_batch_size=512,
            hidden_size=hidden_size,
            dnn_base_units_size_alice=[256, hidden_size],
            dnn_base_units_size_bob=[256, hidden_size],
            dnn_fuse_units_size=[64],
            deepfm_embedding_dim=4,
        )

    def dnn_base_units_size_range_alice(self):
        return [
            [256, 128, -1],
            [256, -1],
            [-1],
        ]

    def dnn_fuse_units_size_range(self):
        return [[64], [64, 64]]

    def deepfm_embedding_dim_range(self):
        return [8, 16]

    def create_base_model_alice(self):
        return TorchModel(
            model_fn=DeepFMBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam, lr=1e-3),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=self.alice_input_dims,
            dnn_units_size=self.dnn_base_units_size_alice,
            continuous_feas_index=self.alice_dense_indexes,
            fm_embedding_dim=self.deepfm_embedding_dim,
        )

    def create_base_model_bob(self):
        return TorchModel(
            model_fn=DeepFMBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam, lr=1e-3),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=self.bob_input_dims,
            dnn_units_size=self.dnn_base_units_size_bob,
            continuous_feas_index=self.bob_dense_indexes,
            fm_embedding_dim=self.deepfm_embedding_dim,
        )

    def create_fuse_model(self):
        return TorchModel(
            model_fn=DeepFMFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam, lr=1e-3),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(Precision, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=self.dnn_fuse_units_size,
        )

    def support_attacks(self):
        return ['norm', 'replay', 'replace']
