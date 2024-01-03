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

from benchmark_examples.autoattack.applications.base import TrainBase
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.utils.simulation.datasets import load_mnist


class MnistBase(TrainBase):
    def __init__(self, config, alice, bob, epoch=1, train_batch_size=128):
        super().__init__(
            config, alice, bob, bob, 10, epoch=epoch, train_batch_size=train_batch_size
        )

    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        sl_model = SLModel(
            base_model_dict={
                self.alice: self.alice_base_model,
                self.bob: self.bob_base_model,
            },
            device_y=self.device_y,
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
            dataset_builder=None,
        )
        logging.warning(history)

    def _prepare_data(self):
        (train_data, train_label), (test_data, test_label) = load_mnist(
            parts={self.alice: (0, 2000), self.bob: (0, 2000)},
            normalized_x=True,
            categorical_y=True,
            is_torch=True,
        )
        return (
            train_data.astype(np.float32),
            train_label.astype(np.float32),
            test_data.astype(np.float32),
            test_label.astype(np.float32),
        )
