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
        if callbacks is None:
            self.callbacks: List[Callback] = []
        else:
            self.callbacks: List[Callback] = (
                callbacks if isinstance(callbacks, List) else [callbacks]
            )

        # callbacks status
        self.stop_training = [False]
        self.history = {}
        if isinstance(callbacks, CallbackList):
            raise RuntimeError("Cannot set a CallbackList to CallbaskList.")

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

    def before_agglayer(self):
        for callback in self.callbacks:
            callback.before_agglayer()

    def after_agglayer(self):
        for callback in self.callbacks:
            callback.after_agglayer()

    def on_before_base_forward(self):
        for callback in self.callbacks:
            callback.on_before_base_forward()

    def on_after_base_forward(self):
        for callback in self.callbacks:
            callback.on_after_base_forward()

    def on_before_fuse_net(self):
        for callback in self.callbacks:
            callback.on_before_fuse_net()

    def on_after_fuse_net(self):
        for callback in self.callbacks:
            callback.on_after_fuse_net()
