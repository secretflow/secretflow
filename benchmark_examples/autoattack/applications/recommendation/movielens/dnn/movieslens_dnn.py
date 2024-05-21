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

from typing import List, Optional

import torch.nn as nn
import torch.optim
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack.applications.recommendation.movielens.movielens_base import (
    MovielensBase,
)
from secretflow.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper


class MovielensDnn(MovielensBase):
    def __init__(self, config, alice, bob, hidden_size=64):
        super().__init__(
            config,
            alice,
            bob,
            epoch=4,
            train_batch_size=128,
            hidden_size=hidden_size,
            dnn_base_units_size_alice=[256, hidden_size],
            dnn_base_units_size_bob=[256, hidden_size],
            dnn_fuse_units_size=[1],
            dnn_embedding_dim=16,
        )

    def create_base_model_alice(self):
        return TorchModel(
            model_fn=DnnBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(Precision, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=self.alice_input_dims,
            dnn_units_size=self.dnn_base_units_size_alice,
            embedding_dim=self.dnn_embedding_dim,
            sparse_feas_indexes=[i for i in range(self.alice_fea_nums)],
        )

    def create_base_model_bob(self):
        return TorchModel(
            model_fn=DnnBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(Precision, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=self.bob_input_dims,
            dnn_units_size=self.dnn_base_units_size_bob,
            embedding_dim=self.dnn_embedding_dim,
            sparse_feas_indexes=[i for i in range(self.bob_fea_nums)],
        )

    def create_fuse_model(self):
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
            dnn_units_size=self.dnn_fuse_units_size,
            output_func=nn.Sigmoid,
        )

    def hidden_size_range(self) -> list:
        return [64, 128]

    def dnn_base_units_size_range_alice(self) -> Optional[List[List[int]]]:
        return [[256, -1], [256, 128, -1]]

    def dnn_base_units_size_range_bob(self) -> Optional[list]:
        return None

    def dnn_fuse_units_size_range(self) -> Optional[list]:
        return [[1], [256, 128, 1]]

    def dnn_embedding_dim_range(self) -> Optional[List[int]]:
        return [8]
