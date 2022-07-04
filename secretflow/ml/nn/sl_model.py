#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# Copyright 2022 Ant Group Co., Ltd.
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


"""SLModel

"""

import math
import os
import secrets
from typing import Callable, Dict, List, Union

import tensorflow as tf
import logging
from tqdm import tqdm

from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, Device, reveal, wait
from secretflow.ml.nn.sl_base import PYUSLTFModel
from secretflow.security.privacy import DPStrategy


class SLModelTF:
    def __init__(
        self,
        base_model_dict: Dict[Device, Callable[[], tf.keras.Model]] = {},
        device_y: PYU = None,
        model_fuse: Callable[[], tf.keras.Model] = None,
        dp_strategy_dict: Dict[Device, DPStrategy] = None,
    ):

        self._workers = {
            device: PYUSLTFModel(
                device=device,
                builder_base=model,
                builder_fuse=None if device != device_y else model_fuse,
                dp_strategy=dp_strategy_dict.get(device, None)
                if dp_strategy_dict
                else None,
            )
            for device, model in base_model_dict.items()
        }
        self.device_y = device_y
        self.dp_strategy_dict = dp_strategy_dict

    def handle_data(
        self,
        x: Union[FedNdarray, VDataFrame, List],
        y: Union[FedNdarray, VDataFrame] = None,
        sample_weight: Union[FedNdarray, VDataFrame] = None,
        batch_size=32,
        shuffle=False,
        epochs=1,
        stage="train",
        random_seed=1234,
    ):
        # Convert VDataFrame to FedNdarray
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        if isinstance(sample_weight, VDataFrame):
            sample_weight = sample_weight.values
        if isinstance(x, List):
            if isinstance(x[0], VDataFrame):
                x = [xi.values for xi in x]
        # training steps_per_epoch
        if isinstance(x, FedNdarray):
            parties_length = x.length()
        elif isinstance(x, List):
            parties_length = x[0].length()
        else:
            raise ValueError(f"Only can be FedNdarray or List, but got {type(x)} ")
        lengths = [length for device, length in parties_length.items()]
        assert len(set(lengths)) == 1, "length of all parties must be same"
        steps_per_epoch = math.ceil(lengths[0] / batch_size)
        for device, worker in self._workers.items():
            if device == self.device_y and y is not None:
                y_partitions = y.partitions[device]
                if sample_weight is not None:
                    s_w_partitions = sample_weight.partitions[device]
                else:
                    s_w_partitions = None
            else:
                y_partitions = None
                s_w_partitions = None
            if isinstance(x, List):
                worker.build_dataset(
                    *[xi.partitions[device] for xi in x],
                    y=y_partitions,
                    s_w=s_w_partitions,
                    batch_size=batch_size,
                    buffer_size=batch_size * 8,
                    shuffle=shuffle,
                    repeat_count=epochs,
                    stage=stage,
                    random_seed=random_seed,
                )
            else:
                worker.build_dataset(
                    *[x.partitions[device]],
                    y=y_partitions,
                    s_w=s_w_partitions,
                    batch_size=batch_size,
                    buffer_size=batch_size * 8,
                    shuffle=shuffle,
                    repeat_count=epochs,
                    stage=stage,
                    random_seed=random_seed,
                )
        return steps_per_epoch

    def fit(
        self,
        x: Union[VDataFrame, FedNdarray, List],
        y: Union[FedNdarray, VDataFrame],
        batch_size=32,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_data=None,
        shuffle=False,
        sample_weight=None,
        validation_freq=1,
        dp_spent_step_freq=None,
    ):
        """Vertical split learning training interface

        Args:
                x: feature, FedNdArray or HDataFrame
                y: label, FedNdArray or HDataFrame
                batch_size: Number of samples per gradient update, Int
                epochs: Number of epochs to train the model
                verbose: 0, 1. Verbosity mode
                callbacks: List of `keras.callbacks.Callback` instances.
                validation_data: Data on which to validate
                shuffle: Whether shuffle dataset or not
                sample_weight: weights for the training samples
                validation_freq: specifies how many training epochs to run before a new validation run is performed
                dp_spent_step_freq: specifies how many training steps to check the budget of dp
        """
        random_seed = secrets.randbelow(100000)
        logging.info(
            f"SL Train Params: batch_size={batch_size} epochs={epochs} shuffle={shuffle} random_seed={random_seed} validation_freq={validation_freq} dp_spent_step_freq={dp_spent_step_freq}"
        )
        # sanity check
        assert isinstance(validation_freq, int) and validation_freq >= 1
        assert len(self._workers) == 2, "split learning only support 2 parties"
        assert isinstance(validation_freq, int) and validation_freq >= 1
        if dp_spent_step_freq is not None:
            assert isinstance(dp_spent_step_freq, int) and dp_spent_step_freq >= 1

        # build dataset
        train_x, train_y = x, y
        if validation_data is not None:
            logging.debug("validation_data provided")
            if len(validation_data) == 2:
                valid_x, valid_y = validation_data
                valid_sample_weight = None
            else:
                valid_x, valid_y, valid_sample_weight = validation_data
        else:
            valid_x, valid_y, valid_sample_weight = None, None, None
        steps_per_epoch = self.handle_data(
            train_x,
            train_y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
            epochs=epochs,
            stage="train",
            random_seed=random_seed,
        )
        validation = False

        if valid_x is not None and valid_y is not None:
            validation = True
            valid_steps = self.handle_data(
                valid_x,
                valid_y,
                sample_weight=valid_sample_weight,
                batch_size=batch_size,
                epochs=epochs,
                stage="eval",
            )

        self._workers[self.device_y].init_training(callbacks, epochs=epochs)
        self._workers[self.device_y].on_train_begin()
        for epoch in range(epochs):
            report_list = []
            report_list.append(f"epoch: {epoch}/{epochs} - ")
            if verbose == 1:
                pbar = tqdm(total=steps_per_epoch)
            self._workers[self.device_y].on_epoch_begin(epoch)
            for step in range(0, steps_per_epoch):
                if verbose == 1:
                    pbar.update(1)
                hiddens = []
                self._workers[self.device_y].on_train_batch_begin(step=step)
                for device, worker in self._workers.items():
                    hidden = worker.base_forward(stage="train")
                    hiddens.append(hidden.to(self.device_y))

                gradients = self._workers[self.device_y].fuse_net(*hiddens)

                idx = 0
                for device, worker in self._workers.items():
                    gradient = gradients[idx].to(device)
                    worker.base_backward(gradient)
                    idx += 1
                self._workers[self.device_y].on_train_batch_end(step=step)

                if dp_spent_step_freq is not None:
                    current_step = epoch * steps_per_epoch + step
                    if current_step % dp_spent_step_freq == 0:
                        privacy_device = {}
                        for device, dp_strategy in self.dp_strategy_dict.items():
                            privacy_dict = dp_strategy.get_privacy_spent(current_step)
                            privacy_device[device] = privacy_dict

            if validation and epoch % validation_freq == 0:
                # validation
                self._workers[self.device_y].reset_metrics()
                for step in range(0, valid_steps):
                    hiddens = []  # driver端
                    for device, worker in self._workers.items():
                        hidden = worker.base_forward("eval")
                        hiddens.append(hidden.to(self.device_y))
                    metrics = self._workers[self.device_y].evaluate(*hiddens)
                self._workers[self.device_y].on_validation(metrics)
            epoch_log = self._workers[self.device_y].on_epoch_end(epoch)
            for name, metric in reveal(epoch_log).items():
                report_list.append(f"{name}:{metric} ")
            report = " ".join(report_list)
            if verbose == 1:
                pbar.set_postfix_str(report)
                pbar.close()
            if reveal(self._workers[self.device_y].get_stop_training()):
                break
        history = self._workers[self.device_y].on_train_end()
        return reveal(history)

    @reveal
    def predict(self, x: Union[FedNdarray, List[FedNdarray]], batch_size=32, verbose=0):
        """Vertical split learning offline prediction interface

        Args:
               x: feature, FedNdArray or HDataFrame
               batch_size: Number of samples per gradient update, Int
               verbose: 0, 1. Verbosity mode
        """
        predict_steps = self.handle_data(
            x, None, batch_size=batch_size, stage="eval", epochs=1
        )
        if verbose > 0:
            pbar = tqdm(total=predict_steps)
            pbar.set_description('Predict Processing:')
        for step in range(0, predict_steps):
            hiddens = []
            for device, worker in self._workers.items():
                hidden = worker.base_forward(stage="eval")
                hiddens.append(hidden.to(self.device_y))
            if verbose > 0:
                pbar.update(1)
        y_pred = self._workers[self.device_y].predict(*hiddens)
        return y_pred

    @reveal
    def evaluate(
        self,
        x: Union[FedNdarray, List[FedNdarray]],
        y: FedNdarray = None,
        batch_size: int = 32,
        sample_weight=None,
        verbose=1,
        steps=None,
    ):
        """Vertical split learning evaluate interface

        Args:
            x: Input data. It could be:
                - FedNdArray
                - HDataFrame
            y: Label. It could be:
                - FedNdArray
                - HDataFrame
            batch_size: Integer or `Dict`. Number of samples per batch of
                computation. If unspecified, `batch_size` will default to 32.
            sample_weight: Optional Numpy array of weights for the test samples,
                used for weighting the loss function.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
            steps: Integer or `None`. Total number of steps (batches of samples)
        Returns:
            metrics: federate evaluate result
        """

        evaluate_steps = self.handle_data(
            x,
            y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            stage="eval",
            epochs=1,
        )
        metrics = None
        self._workers[self.device_y].reset_metrics()
        if verbose > 0:
            pbar = tqdm(total=evaluate_steps)
            pbar.set_description('Evaluate Processing:')
        for step in range(0, evaluate_steps):
            hiddens = []  # driver端
            for device, worker in self._workers.items():
                hidden = worker.base_forward(stage="eval")
                hiddens.append(hidden.to(self.device_y))
            if verbose > 0:
                pbar.update(1)
            metrics = self._workers[self.device_y].evaluate(*hiddens)
        report_list = [f"{k}:{v}" for k, v in reveal(metrics).items()]
        report = " ".join(report_list)
        if verbose == 1:
            pbar.set_postfix_str(report)
            pbar.close()
        return metrics

    def save_model(
        self,
        base_model_path: Union[str, Dict[PYU, str]] = None,
        fuse_model_path: str = None,
        is_test=False,
    ):
        """Vertical split learning save model interface

        Args:
               base_model_path: base model path
               fuse_model_path: fuse model path
               is_test: whether is test mode
        """
        assert isinstance(
            base_model_path, (str, Dict)
        ), f'Model path accepts string or dict but got {type(base_model_path)}.'
        assert fuse_model_path is not None, "Fuse model path cannot be empty"
        if isinstance(base_model_path, str):
            base_model_path = {
                device: base_model_path for device in self._workers.keys()
            }

        res = []
        for device, worker in self._workers.items():
            assert (
                device in base_model_path
            ), f'Should provide a path for device {device}.'
            if is_test:
                base_model_path_test = os.path.join(
                    base_model_path[device], device.__str__().strip("_")
                )
                res.append(worker.save_base_model(base_model_path_test))
            else:
                res.append(worker.save_base_model(base_model_path[device]))
        res.append(self._workers[self.device_y].save_fuse_model(fuse_model_path))
        wait(res)

    def load_model(
        self,
        base_model_path: Union[str, Dict[PYU, str]] = None,
        fuse_model_path: str = None,
        is_test=False,
    ):
        """Vertical split learning load model interface

        Args:
               base_model_path: base model path
               fuse_model_path: fuse model path
               is_test: whether is test mode
        """
        assert isinstance(
            base_model_path, (str, Dict)
        ), f'Model path accepts string or dict but got {type(base_model_path)}.'
        assert fuse_model_path is not None, "Fuse model path cannot be empty"
        if isinstance(base_model_path, str):
            base_model_path = {
                device: base_model_path for device in self._workers.keys()
            }

        res = []
        for device, worker in self._workers.items():
            assert (
                device in base_model_path
            ), f'Should provide a path for device {device}.'
            if is_test:
                # only execute when unittest
                base_model_path_test = os.path.join(
                    base_model_path[device], device.__str__().strip("_")
                )
                res.append(worker.load_base_model(base_model_path_test))
            else:
                res.append(worker.load_base_model(base_model_path))
        res.append(self._workers[self.device_y].load_fuse_model(fuse_model_path))
        wait(res)
