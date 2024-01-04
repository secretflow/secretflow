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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from secretflow.ml.nn.callbacks.callback import Callback


class TrainBase(ABC):
    def __init__(
        self,
        config: Dict,
        alice,
        bob,
        device_y,
        num_classes,
        epoch=2,
        train_batch_size=128,
    ):
        self.config = config
        self.alice = alice
        self.epoch = config.get('epoch', epoch)
        self.train_batch_size = config.get('train_batch_size', train_batch_size)
        self.bob = bob
        self.device_y = device_y
        self.num_classes = num_classes
        self.config = config
        (
            self.train_data,
            self.train_label,
            self.test_data,
            self.test_label,
        ) = self._prepare_data()
        self.alice_base_model = self._create_base_model_alice()
        self.bob_base_model = self._create_base_model_bob()
        self.fuse_model = self._create_fuse_model()

    @abstractmethod
    def train(self, callbacks: Optional[Union[List[Callback], Callback]] = None):
        pass

    def predict(self):
        raise NotImplementedError("Predict not implemented.")

    @abstractmethod
    def _prepare_data(self):
        pass

    @abstractmethod
    def _create_base_model_alice(self):
        pass

    @abstractmethod
    def _create_base_model_bob(self):
        pass

    @abstractmethod
    def _create_fuse_model(self):
        pass

    def support_attacks(self):
        """
        Which attacks this application supports.
        Returns:
            List of attack names, default is empty.
        """
        return []

    def lia_auxiliary_data_builder(self, batch_size=16, file_path=None):
        raise NotImplementedError(
            f"need implement lia_auxiliary_data_builder on {type(self).__name__} "
        )

    def lia_auxiliary_model(self, ema=False):
        raise NotImplementedError(
            f"need implement lia_auxiliary_model on {type(self).__name__} "
        )

    def fia_auxiliary_data_builder(self):
        raise NotImplementedError(
            f"need implement fia_auxiliary_data_builder on {type(self).__name__} "
        )
