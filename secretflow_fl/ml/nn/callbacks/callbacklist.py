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

from typing import List

from .callback import Callback
from .early_stopping import EarlyStoppingBase
from .history import History
from .progbar import Progbar


class CallbackList:
    """Container abstraction a list of callback on driver side"""

    def __init__(
        self,
        callbacks=None,
        workers=None,
        device_y=None,
        add_history=False,
        add_progbar=False,
        **kwargs,
    ):
        if isinstance(callbacks, CallbackList):
            raise RuntimeError("Cannot set a CallbackList to CallbaskList.")
        if callbacks is None:
            self.callbacks: List[Callback] = []
        elif isinstance(callbacks, List):
            self.callbacks = callbacks.copy()
        else:
            self.callbacks = [callbacks]

        # callbacks status
        self.stop_training = [False]
        self.history = {}

        if add_progbar:
            self.callbacks.append(Progbar())
        if add_history:
            self.callbacks.append(
                History(
                    history=self.history,
                )
            )
        # for early stopping
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingBase):
                callback.set_stop_training(self.stop_training)

        self.set_workers(workers, device_y)

        if kwargs:
            self.set_params(kwargs)

    def set_workers(self, workers, device_y):
        for callback in self.callbacks:
            callback.set_workers(
                workers,
                device_y,
            )

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_test_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_begin(logs=logs)

    def on_test_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_end(logs=logs)

    def on_predict_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_predict_begin()

    def on_predict_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch=0, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch=0):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch=0):
        for callback in self.callbacks:
            callback.on_batch_end(batch)

    def on_train_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch)

    def on_train_batch_end(self, batch):
        for callback in self.callbacks:
            callback.on_train_batch_end(batch)

    def on_test_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch)

    def on_test_batch_end(self, batch):
        for callback in self.callbacks:
            callback.on_test_batch_end(batch)

    def on_predict_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_predict_batch_begin(batch)

    def on_predict_batch_end(self, batch):
        for callback in self.callbacks:
            callback.on_predict_batch_end(batch)

    def on_agglayer_forward_begin(self, hiddens=None):
        for callback in self.callbacks:
            callback.on_agglayer_forward_begin(hiddens=hiddens)

    def on_agglayer_forward_end(self, hiddens=None):
        for callback in self.callbacks:
            callback.on_agglayer_forward_end(hiddens=hiddens)

    def on_agglayer_backward_begin(self, gradients=None):
        for callback in self.callbacks:
            callback.on_agglayer_backward_begin(gradients=gradients)

    def on_agglayer_backward_end(self, gradients=None):
        for callback in self.callbacks:
            callback.on_agglayer_backward_end(gradients=gradients)

    def on_base_forward_begin(self):
        for callback in self.callbacks:
            callback.on_base_forward_begin()

    def on_base_forward_end(self):
        for callback in self.callbacks:
            callback.on_base_forward_end()

    def on_base_backward_begin(self):
        for callback in self.callbacks:
            callback.on_base_backward_begin()

    def on_base_backward_end(self):
        for callback in self.callbacks:
            callback.on_base_backward_end()

    def on_fuse_forward_begin(self):
        for callback in self.callbacks:
            callback.on_fuse_forward_begin()

    def on_fuse_forward_end(self):
        for callback in self.callbacks:
            callback.on_fuse_forward_end()

    def on_fuse_backward_begin(self):
        for callback in self.callbacks:
            callback.on_fuse_backward_begin()

    def on_fuse_backward_end(self):
        for callback in self.callbacks:
            callback.on_fuse_backward_end()
