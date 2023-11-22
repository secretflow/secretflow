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
from tqdm import tqdm

import secretflow as sf
from secretflow.device import reveal
from secretflow.ml.nn.metrics import aggregate_metrics
from .callback import Callback


class Progbar(Callback):
    """Callback that prints metrics to stdout."""

    def __init__(self):
        super(Progbar, self).__init__()
        self.progbar = None
        self.target = None
        self.verbose = 1
        self.epochs = 1
        self.steps = 0
        self.epoch_logs = {}
        self._train_step, self._test_step, self._predict_step = None, None, None
        self._called_in_fit = False

    def set_params(self, params):
        self.verbose = params.get('verbose', 0)
        self.target = params.get('steps', None)
        self.epochs = params.get('epochs', 1)

    def on_train_begin(self, logs=None):
        # When this logger is called inside `fit`, validation is silent.
        self._called_in_fit = True

    def on_test_begin(self, logs=None):
        if not self._called_in_fit:
            self._reset_progbar(self.target)
            self.progbar.set_description("Evaluate Processing: ")

    def on_predict_begin(self):
        self._reset_progbar(self.target)
        self.progbar.set_description("Predict Processing: ")

    def on_epoch_begin(self, epoch):
        self._reset_progbar(self.target)

        if self.verbose and self.epochs > 1:
            self.progbar.set_description("Train Processing: ")
            print('Epoch %d/%d' % (epoch + 1, self.epochs))

    def on_train_batch_end(self, batch):
        update = batch - self.steps
        if self.progbar:
            self.progbar.update(update)
        self.steps = batch

    def on_test_batch_end(self, batch):
        if not self._called_in_fit and self.progbar:
            update = batch - self.steps
            self.progbar.update(update)
            self.steps = batch

    def on_predict_batch_end(self, batch):
        update = batch - self.steps
        if self.progbar:
            self.progbar.update(update)
        self.steps = batch

    def on_epoch_end(self, epoch=None, logs=None):
        # epoch end output report to logger
        if self.device_y:
            # deal with split learning
            report = reveal(self._workers[self.device_y].get_logs())
        else:
            # deal with federated learning
            local_metrics = []
            for device, worker in self._workers.items():
                _metrics = worker.get_local_metrics()
                local_metrics.append(_metrics)

            metrics = aggregate_metrics(local_metrics=reveal(local_metrics))

            report = {}
            for m in metrics:
                report[m.name] = m.result().numpy()
        if self.verbose == 1:
            self.progbar.set_postfix_str(report)
            self.progbar.close()

    def on_test_end(self, logs=None):
        # output report and close pbar
        if not self._called_in_fit:
            report = sf.reveal(logs)
            if self.verbose == 1:
                self.progbar.set_postfix(report)
                self.progbar.close()

    def on_predict_end(self, logs=None):
        if self.progbar:
            self.progbar.close()

    def _reset_progbar(self, target):
        self.steps = 0
        self.progbar = tqdm(total=target) if self.verbose == 1 else None
