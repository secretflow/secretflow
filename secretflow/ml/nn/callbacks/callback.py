# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional

from secretflow import PYU


class Callback:
    """Abstract base class used to build new callbacks on driver side"""

    def __init__(self, **kwargs):
        self.params = kwargs
        self._workers: Optional[Dict] = None
        self.device_y: Optional[PYU] = None

    def set_workers(self, workers, device_y):
        self._workers = workers
        self.device_y = device_y

    def set_params(self, params):
        self.params.update(params)

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch=None, logs=None):
        pass

    def on_epoch_end(self, epoch=None, logs=None):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    def on_train_batch_begin(self, batch):
        pass

    def on_train_batch_end(self, batch):
        pass

    def on_test_batch_begin(self, batch):
        pass

    def on_test_batch_end(self, batch):
        pass

    def on_predict_batch_begin(self, batch):
        pass

    def on_predict_batch_end(self, batch):
        pass

    def before_agglayer(self):
        pass

    def after_agglayer(self):
        pass

    def on_before_base_forward(self):
        pass

    def on_after_base_forward(self):
        pass

    def on_before_fuse_net(self):
        pass

    def on_after_fuse_net(self):
        pass
