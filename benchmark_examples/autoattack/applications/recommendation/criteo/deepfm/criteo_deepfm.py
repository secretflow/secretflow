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

alice_sparse_features = ['C' + str(i) for i in range(1, 27)]
bob_dense_features = ['I' + str(i) for i in range(1, 14)]


class CriteoDeepfm(CriteoBase):
    def __init__(self, config, alice, bob):
        super().__init__(
            config, alice, bob, epoch=3, train_batch_size=512, hidden_size=64
        )

    def _create_base_model_alice(self):
        model = TorchModel(
            model_fn=DeepFMBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam, lr=1e-2),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=self.alice_input_dims,
            dnn_units_size=[200, 100, self.hidden_size],
            continuous_feas_index=self.alice_dense_indexes,
        )
        return model

    def _create_base_model_bob(self):
        model = TorchModel(
            model_fn=DeepFMBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam, lr=1e-2),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=self.bob_input_dims,
            dnn_units_size=[200, 100, self.hidden_size],
            continuous_feas_index=self.bob_dense_indexes,
        )
        return model

    def _create_fuse_model(self):
        return TorchModel(
            model_fn=DeepFMFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam, lr=1e-2),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                metric_wrapper(Precision, task="binary"),
                metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[self.hidden_size, self.hidden_size],
            dnn_units_size=[64, 1],
        )
