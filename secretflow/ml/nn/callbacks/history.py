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
from secretflow.device import reveal
from .callback import Callback
from secretflow.ml.nn.metrics import aggregate_metrics


class History(Callback):
    """Callback that records events into a `History` object."""

    def __init__(
        self,
        history=None,
        **params,
    ):
        super().__init__(
            **params,
        )
        self.history = history

    def on_train_begin(self, logs=None):
        self.epoch = []
        if self.device_y is None:
            self.history["global_history"] = {}
            self.history["local_history"] = {}

    def on_epoch_end(self, epoch, logs=None):
        if self.device_y:
            metrics = reveal(self._workers[self.device_y].get_logs())
            self.epoch.append(epoch)
            for k, v in metrics.items():
                self.history.setdefault(k, []).append(v)
        else:
            # deal with federated learning
            self.epoch.append(epoch)
            local_metrics = []
            for device, worker in self._workers.items():
                _metrics = worker.get_local_metrics()
                local_metrics.append(_metrics)

            metrics = aggregate_metrics(local_metrics=reveal(local_metrics))

            global_history = {}
            for m in metrics:
                global_history[m.name] = m.result().numpy()
            for k, v in global_history.items():
                self.history["global_history"].setdefault(k, []).append(v)
            for device, worker in self._workers.items():
                local_history = reveal(worker.get_logs())
                for k, v in local_history.items():
                    self.history["local_history"].setdefault(
                        f"{device}_{k}", []
                    ).append(v)
