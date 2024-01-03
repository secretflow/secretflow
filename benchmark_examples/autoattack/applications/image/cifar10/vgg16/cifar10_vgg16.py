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

from torch import nn, optim
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack.applications.image.cifar10.cifar10_base import (
    Cifar10TrainBase,
)
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.applications.sl_vgg_torch import VGGBase, VGGFuse
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper


class Cifar10VGG16(Cifar10TrainBase):
    def __init__(self, config, alice, bob):
        super().__init__(config, alice, bob)

    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        logging.warning(
            f"the batchsize = {self.train_batch_size}, epoch = {self.epoch}"
        )
        device_y = self.bob
        sl_model = SLModel(
            base_model_dict={
                self.alice: self.alice_base_model,
                self.bob: self.bob_base_model,
            },
            device_y=device_y,
            model_fuse=self.fuse_model,
            simulation=True,
            random_seed=1234,
            backend='torch',
            strategy='split_nn',
        )
        history = sl_model.fit(
            x=self.train_data,
            y=self.train_label,
            validation_data=(self.test_data, self.test_label),
            epochs=self.epoch,
            batch_size=self.train_batch_size,
            shuffle=False,
            random_seed=1234,
        )
        logging.warning(history)

    def _create_base_model(self):
        return TorchModel(
            model_fn=VGGBase,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-4),
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
        )

    def _create_base_model_alice(self):
        return self._create_base_model()

    def _create_base_model_bob(self):
        return self._create_base_model()

    def _create_fuse_model(self):
        return TorchModel(
            model_fn=VGGFuse,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-4),
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
        )
