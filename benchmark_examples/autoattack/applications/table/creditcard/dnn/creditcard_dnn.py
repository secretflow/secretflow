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
from torchmetrics import Accuracy

from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.applications.sl_dnn_torch import DnnBase, DnnFuse
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper
from secretflow.preprocessing import StandardScaler
from secretflow.utils.simulation.datasets import load_creditcard


class CreditcardDnn(TrainBase):
    def __init__(self, config, alice, bob):
        super().__init__(config, alice, bob, bob, 2, epoch=2, train_batch_size=1024)

    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        base_model_dict = {
            self.alice: self.alice_base_model,
            self.bob: self.bob_base_model,
        }
        # Define DP operations
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
        data = load_creditcard({self.alice: (0, 25), self.bob: (25, 29)})
        label = load_creditcard({self.bob: (29, 30)}).astype(np.float32)
        scaler = StandardScaler()
        data = scaler.fit_transform(data).astype(np.float32)
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
            model_fn=DnnBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
                # metric_wrapper(Precision, task="binary"),
                # metric_wrapper(AUROC, task="binary"),
            ],
            input_dims=[25],
            dnn_units_size=[int(28 / 2), 28],
        )
        return model

    def _create_base_model_bob(self):
        model = TorchModel(
            model_fn=DnnBase,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
            ],
            input_dims=[4],
            dnn_units_size=[4],
        )
        return model

    def _create_fuse_model(self):
        model = TorchModel(
            model_fn=DnnFuse,
            loss_fn=nn.BCELoss,
            optim_fn=optim_wrapper(torch.optim.Adam),
            metrics=[
                metric_wrapper(Accuracy, task="binary"),
            ],
            input_dims=[28, 4],
            dnn_units_size=[1],
        )
        return model

    def support_attacks(self):
        return ['norm']
