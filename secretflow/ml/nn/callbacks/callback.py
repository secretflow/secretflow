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

from secretflow.device import PYU
from secretflow.device import reveal
from secretflow.ml.nn.metrics import aggregate_metrics


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

    def on_agglayer_forward_begin(self, hiddens=None):
        pass

    def on_agglayer_forward_end(self, hiddens=None):
        pass

    def on_agglayer_backward_begin(self, gradients=None):
        pass

    def on_agglayer_backward_end(self, gradients=None):
        pass

    def on_base_forward_begin(self):
        pass

    def on_base_forward_end(self):
        pass

    def on_base_backward_begin(self):
        pass

    def on_base_backward_end(self):
        pass

    def on_fuse_forward_begin(self):
        pass

    def on_fuse_forward_end(self):
        """this hook is not working for now."""
        pass

    def on_fuse_backward_begin(self):
        """this hook is not working for now."""
        pass

    def on_fuse_backward_end(self):
        pass

    def _calc_aggregated_metrics(self):
        """
        Collect metrics from each worker and calculate aggregated metrics.
        Just need to calculate result of the last aggregated metric,
        as later calculated result would overwrite previous ones.
        This operation will greatly speedup metrics calculation.
        """
        assert len(self._workers) > 0, "num of worker should be greater than zero"
        local_metrics = []
        for _, worker in self._workers.items():
            _metrics = worker.get_local_metrics()
            local_metrics.append(_metrics)

        aggregated_metrics = aggregate_metrics(local_metrics=reveal(local_metrics))

        # keep the last metric for each metric name.
        metrics_dict = {m.name: m for m in aggregated_metrics}

        return {m.name: m.result().numpy() for m in metrics_dict.values()}
