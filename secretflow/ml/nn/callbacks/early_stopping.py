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
import logging
import numpy as np
from typing import List
from secretflow.ml.nn.callbacks.callback import Callback
from secretflow.device import reveal


class EarlyStoppingBase(Callback):
    """Stop training when a monitored metric has stopped improving.
    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch or
    every check step whether the loss is no longer decreasing,
    considering the `min_delta` and `patience` if applicable. Once
    it's found no longer decreasing, `model.stop_training` is marked True
    and the training terminates. The quantity to be monitored needs to be
    available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.
    Args:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `"max"`
            mode it will stop when the quantity
            monitored has stopped increasing; in `"auto"`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
    """

    def __init__(
        self,
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto',
        baseline=None,
    ):
        super(EarlyStoppingBase, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.best_weights = None
        if mode not in ['auto', 'min', 'max']:
            logging.warning(
                'EarlyStopping mode %s is unknown, ' 'fallback to auto mode.', mode
            )
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or 'auc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.stop_training = None

    def set_stop_training(self, stop_training: List[bool]):
        self.stop_training = stop_training

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def on_test_end(self, logs):
        self.val_metrics = reveal(logs)

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s',
                self.monitor,
                ','.join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


class EarlyStoppingBatch(EarlyStoppingBase):
    """Stop training when a monitored metric has stopped improving in one epoch,
    the metric will be evaluated every check_step.
    Args:
        check_step: Every check_step the model will be evaluated and update metric
        warm_up_step: Before warm_up_step, this callback will not be effective
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `"max"`
            mode it will stop when the quantity
            monitored has stopped increasing; in `"auto"`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
    """

    def __init__(
        self,
        check_step,
        warm_up_step=0,
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto',
        baseline=None,
    ):
        super(EarlyStoppingBatch, self).__init__(
            monitor, min_delta, patience, verbose, mode, baseline
        )
        self.check_step = check_step
        self.warm_up_step = warm_up_step
        self.stopped_batch = 0

    def on_train_batch_end(self, step):
        if step <= self.warm_up_step or step % self.check_step != 0:
            return
        current = self.get_monitor_value(self.val_metrics)
        if current is None:
            return
        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0
        if self.wait >= self.patience:
            self.stopped_batch = step
            self.stop_training[0] = True

    def on_train_end(self, logs=None):
        if self.stopped_batch > 0 and self.verbose > 0:
            logging.info('Batch %05d: early stopping' % (self.stopped_batch + 1))


class EarlyStoppingEpoch(EarlyStoppingBase):
    """Stop training when a monitored metric has stopped improving,
    the metric will be evaluated every epoch.
    Args:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `"max"`
            mode it will stop when the quantity
            monitored has stopped increasing; in `"auto"`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
    """

    def __init__(
        self,
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto',
        baseline=None,
    ):
        super(EarlyStoppingEpoch, self).__init__(
            monitor, min_delta, patience, verbose, mode, baseline
        )
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(self.val_metrics)
        if current is None:
            return
        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.stop_training[0] = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            logging.info('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
