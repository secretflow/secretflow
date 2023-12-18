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

from typing import Dict

import secretflow.tune as tune
from secretflow.ml.nn.callbacks.attack import AttackCallback
from secretflow.ml.nn.callbacks.callback import Callback


class AutoAttackCallback(Callback):
    """
    AutoAttackCallback is the encapsulation of the AttackCallback, which is used to achieve auto attack.
    When using sf.tune to tune your attack callback, you can use AutoAttackCallback to report attack
    metrics to tuner during the attack.
    Usage:
    .. code-block:: python

        attack_callback = YourAttackCallback()
        attack2_callback
        autoattack = AutoAttackCallback(attack_callback)
        sl_model.fit(
            callback=[attack_callback,attack2_callback,autoattack]
        )
    """

    attack_callback: AttackCallback

    def __init__(self, attack_callback: AttackCallback, **kwargs):
        self.attack_callback = attack_callback
        self.attack_callback.set_workers(self._workers, self.device_y)
        super().__init__(**kwargs)

    def get_final_metrics(self):
        metrics = self.attack_callback.get_attack_metrics()
        if len(metrics) == 0:
            raise RuntimeError(
                f"Auto attack cannot find any attack metrics."
                f"Maybe you did not put any metrics into attack_history in your attack callbacks."
                f"Try use AttackCallback.add_history() after got your metrics."
            )
        return metrics

    def report(self, metrics: Dict):
        assert isinstance(metrics, Dict) and len(metrics) > 0
        tune.train.report(metrics)

    def on_train_begin(self, logs=None):
        self.attack_callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        self.attack_callback.on_train_end(logs)

    def on_predict_begin(self, logs=None):
        self.attack_callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        self.attack_callback.on_predict_end(logs)

    def on_test_begin(self, logs=None):
        self.attack_callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        self.attack_callback.on_test_end(logs)

    def on_epoch_begin(self, epoch=None, logs=None):
        self.attack_callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch=None, logs=None):
        self.attack_callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch):
        self.attack_callback.on_batch_begin(batch)

    def on_batch_end(self, batch):
        self.attack_callback.on_batch_end(batch)

    def on_train_batch_begin(self, batch):
        self.attack_callback.on_train_batch_begin(batch)

    def on_train_batch_end(self, batch):
        self.attack_callback.on_train_batch_end(batch)

    def on_test_batch_begin(self, batch):
        self.attack_callback.on_test_batch_begin(batch)

    def on_test_batch_end(self, batch):
        self.attack_callback.on_test_batch_end(batch)

    def on_predict_batch_begin(self, batch):
        self.attack_callback.on_predict_batch_begin(batch)

    def on_predict_batch_end(self, batch):
        self.attack_callback.on_predict_batch_end(batch)

    def before_agglayer(self):
        self.attack_callback.before_agglayer()

    def after_agglayer(self):
        self.attack_callback.after_agglayer()

    def on_before_base_forward(self):
        self.attack_callback.on_before_base_forward()

    def on_after_base_forward(self):
        self.attack_callback.on_after_base_forward()

    def on_before_fuse_net(self):
        self.attack_callback.on_before_fuse_net()

    def on_after_fuse_net(self):
        self.attack_callback.on_after_fuse_net()
