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

import logging
import math
import os
import secrets
from typing import Callable, Dict, Iterable, List, Tuple, Union

from tqdm import tqdm

from secretflow.data.base import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, Device, reveal, wait
from secretflow.device.device.pyu import PYUObject
from secretflow.ml.nn.sl.backend.tensorflow.sl_base import PYUSLTFModel
from secretflow.security.privacy import DPStrategy
from secretflow.utils.compressor import Compressor


class SLModel:
    def __init__(
        self,
        base_model_dict: Dict[Device, Callable[[], 'tensorflow.keras.Model']] = {},
        device_y: PYU = None,
        model_fuse: Callable[[], 'tensorflow.keras.Model'] = None,
        compressor: Compressor = None,
        dp_strategy_dict: Dict[Device, DPStrategy] = None,
        **kwargs,
    ):

        self.device_y = device_y
        self.dp_strategy_dict = dp_strategy_dict
        self.simulation = kwargs.get('simulation', False)
        self.num_parties = len(base_model_dict)
        self._workers = {
            device: PYUSLTFModel(
                device=device,
                builder_base=model,
                builder_fuse=None if device != device_y else model_fuse,
                compressor=compressor,
                dp_strategy=dp_strategy_dict.get(device, None)
                if dp_strategy_dict
                else None,
            )
            for device, model in base_model_dict.items()
        }

        self.has_compressor = compressor is not None

    def handle_data(
        self,
        x: Union[
            VDataFrame,
            FedNdarray,
            List[Union[HDataFrame, VDataFrame, FedNdarray]],
        ],
        y: Union[FedNdarray, VDataFrame, PYUObject] = None,
        sample_weight: Union[FedNdarray, VDataFrame] = None,
        batch_size=32,
        shuffle=False,
        epochs=1,
        stage="train",
        random_seed=1234,
        dataset_builder: Callable = None,
    ):
        if isinstance(x, (VDataFrame, FedNdarray)):
            x = [x]

        steps_per_epoch = None
        # NOTE: if dataset_builder is set, it should return steps per epoch.
        if dataset_builder is None:
            parties_length = [
                shape[0] for device, shape in x[0].partition_shape().items()
            ]
            assert len(set(parties_length)) == 1, "length of all parties must be same"
            steps_per_epoch = math.ceil(parties_length[0] / batch_size)
            # set steps_per_epoch to device_y
            self._workers[self.device_y].set_steps_per_epoch(steps_per_epoch)

        worker_steps = []
        for device, worker in self._workers.items():
            if device == self.device_y and y is not None:
                if isinstance(y, FedNdarray):
                    y_partitions = y.partitions[device]
                elif isinstance(y, VDataFrame):
                    y_partitions = y.partitions[device].data
                else:
                    assert y.device == device, f"label must be located in device_y"
                    y_partitions = y

                s_w_partitions = (
                    sample_weight.partitions[device]
                    if sample_weight is not None
                    else None
                )
            else:
                y_partitions = None
                s_w_partitions = None

            xs = [xi.partitions[device] for xi in x]
            xs = [t.data if isinstance(t, Partition) else t for t in xs]
            steps = worker.build_dataset(
                *xs,
                y=y_partitions,
                s_w=s_w_partitions,
                batch_size=batch_size,
                buffer_size=batch_size * 8,
                shuffle=shuffle,
                repeat_count=epochs,
                stage=stage,
                random_seed=random_seed,
                dataset_builder=dataset_builder,
            )
            worker_steps.append(steps)

        if dataset_builder is None:
            return steps_per_epoch

        worker_steps = reveal(worker_steps)
        assert (
            len(set(worker_steps)) == 1
        ), "steps_per_epoch of all parties must be same"
        return worker_steps[0]

    def fit(
        self,
        x: Union[
            VDataFrame,
            FedNdarray,
            List[Union[HDataFrame, VDataFrame, FedNdarray]],
        ],
        y: Union[VDataFrame, FedNdarray, PYUObject],
        batch_size=32,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_data=None,
        shuffle=False,
        sample_weight=None,
        validation_freq=1,
        dp_spent_step_freq=None,
        dataset_builder: Callable[[List], Tuple[int, Iterable]] = None,
        audit_log_dir: str = None,
        random_seed: int = None,
    ):
        """Vertical split learning training interface

        Args:
            x: Input data. It could be:

            - VDataFrame: a vertically aligned dataframe.
            - FedNdArray: a vertically aligned ndarray.
            - List[Union[HDataFrame, VDataFrame, FedNdarray]]: list of dataframe or ndarray.

            y: Target data. It could be a VDataFrame or FedNdarray which has only one partition, or a PYUObject.
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train the model
            verbose: 0, 1. Verbosity mode
            callbacks: List of `keras.callbacks.Callback` instances.
            validation_data: Data on which to validate
            shuffle: Whether shuffle dataset or not
            validation_freq: specifies how many training epochs to run before a new validation run is performed
            sample_weight: weights for the training samples
            dp_spent_step_freq: specifies how many training steps to check the budget of dp
            dataset_builder: Callable function, its input is `x` or `[x, y]` if y is set, it should return a
                iterable dataset which should has `steps_per_epoch` property. Dataset builder is mainly for
                building graph dataset.
        """
        if random_seed is None:
            random_seed = secrets.randbelow(100000)

        params = locals()
        logging.info(f"SL Train Params: {params}")
        # sanity check
        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), f"batch_size should be integer > 0"
        assert isinstance(validation_freq, int) and validation_freq >= 1
        assert len(self._workers) == 2, "split learning only support 2 parties"
        assert isinstance(validation_freq, int) and validation_freq >= 1
        if dp_spent_step_freq is not None:
            assert isinstance(dp_spent_step_freq, int) and dp_spent_step_freq >= 1

        # get basenet ouput num
        self.basenet_output_num = {
            device: reveal(worker.get_basenet_output_num())
            for device, worker in self._workers.items()
        }

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
            dataset_builder=dataset_builder,
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
                dataset_builder=dataset_builder,
            )

        self._workers[self.device_y].init_training(callbacks, epochs=epochs)
        self._workers[self.device_y].on_train_begin()

        fuse_net_num_returns = sum(self.basenet_output_num.values())

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
                    # enable compression in fit when model has compressor
                    hidden = worker.base_forward(
                        stage="train", compress=self.has_compressor
                    )
                    hiddens.append(hidden.to(self.device_y))

                gradients = self._workers[self.device_y].fuse_net(
                    *hiddens,
                    _num_returns=fuse_net_num_returns,
                    compress=self.has_compressor,
                )

                idx = 0
                for device, worker in self._workers.items():
                    gradient_list = []
                    for i in range(self.basenet_output_num[device]):
                        gradient = gradients[idx + i].to(device)
                        gradient_list.append(gradient)

                    worker.base_backward(gradient_list, compress=self.has_compressor)
                    idx += self.basenet_output_num[device]
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
                    hiddens = []  # driver end
                    for device, worker in self._workers.items():
                        hidden = worker.base_forward("eval")
                        hiddens.append(hidden.to(self.device_y))
                    metrics = self._workers[self.device_y].evaluate(*hiddens)
                self._workers[self.device_y].on_validation(metrics)

                # save checkpoint
                if audit_log_dir is not None:
                    epoch_base_model_path = os.path.join(
                        audit_log_dir,
                        "base_model",
                        str(epoch),
                    )
                    epoch_fuse_model_path = os.path.join(
                        audit_log_dir,
                        "fuse_model",
                        str(epoch),
                    )
                    self.save_model(
                        base_model_path=epoch_base_model_path,
                        fuse_model_path=epoch_fuse_model_path,
                        is_test=self.simulation,
                        save_traces=True if dataset_builder is None else False,
                    )
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

    def predict(
        self,
        x: Union[
            VDataFrame,
            FedNdarray,
            List[Union[HDataFrame, VDataFrame, FedNdarray]],
        ],
        batch_size=32,
        verbose=0,
        dataset_builder: Callable[[List], Tuple[int, Iterable]] = None,
        compress: bool = False,
    ):
        """Vertical split learning offline prediction interface

        Args:
            x: Input data. It could be:

            - VDataFrame: a vertically aligned dataframe.
            - FedNdArray: a vertically aligned ndarray.
            - List[Union[HDataFrame, VDataFrame, FedNdarray]]: list of dataframe or ndarray.

            batch_size: Number of samples per gradient update, Int
            verbose: 0, 1. Verbosity mode
            dataset_builder: Callable function, its input is `x` or `[x, y]` if y is set, it should return
              steps_per_epoch and iterable dataset. Dataset builder is mainly for building graph dataset.
            compress: Whether to use compressor to compress cross device data.
        """

        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), f"batch_size should be integer > 0"
        if compress:
            assert self.has_compressor, "can not found compressor in model"
        predict_steps = self.handle_data(
            x,
            None,
            batch_size=batch_size,
            stage="eval",
            epochs=1,
            dataset_builder=dataset_builder,
        )
        if verbose > 0:
            pbar = tqdm(total=predict_steps)
            pbar.set_description('Predict Processing:')
        result = []
        for step in range(0, predict_steps):
            hiddens = []
            for device, worker in self._workers.items():
                hidden = worker.base_forward(stage="eval", compress=compress)
                hiddens.append(hidden.to(self.device_y))
            if verbose > 0:
                pbar.update(1)
            y_pred = self._workers[self.device_y].predict(*hiddens, compress=compress)
            result.append(y_pred)
        return result

    @reveal
    def evaluate(
        self,
        x: Union[
            VDataFrame,
            FedNdarray,
            List[Union[HDataFrame, VDataFrame, FedNdarray]],
        ],
        y: Union[VDataFrame, FedNdarray, PYUObject],
        batch_size: int = 32,
        sample_weight=None,
        verbose=1,
        dataset_builder: Callable[[List], Tuple[int, Iterable]] = None,
        random_seed: int = None,
        compress: bool = False,
    ):
        """Vertical split learning evaluate interface

        Args:
            x: Input data. It could be:

            - VDataFrame: a vertically aligned dataframe.
            - FedNdArray: a vertically aligned ndarray.
            - List[Union[HDataFrame, VDataFrame, FedNdarray]]: list of dataframe or ndarray.

            y: Target data. It could be a VDataFrame or FedNdarray which has only one partition, or a PYUObject.
            batch_size: Integer or `Dict`. Number of samples per batch of
                computation. If unspecified, `batch_size` will default to 32.
            sample_weight: Optional Numpy array of weights for the test samples,
                used for weighting the loss function.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
            dataset_builder: Callable function, its input is `x` or `[x, y]` if y is set, it should return
              steps_per_epoch and iterable dataset. Dataset builder is mainly for building graph dataset.
            compress: Whether to use compressor to compress cross device data.
        Returns:
            metrics: federate evaluate result
        """

        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), f"batch_size should be integer > 0"
        if compress:
            assert self.has_compressor, "can not found compressor in model"
        if random_seed is None:
            random_seed = secrets.randbelow(100000)
        evaluate_steps = self.handle_data(
            x,
            y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            stage="eval",
            epochs=1,
            random_seed=random_seed,
            dataset_builder=dataset_builder,
        )
        metrics = None
        self._workers[self.device_y].reset_metrics()
        if verbose > 0:
            pbar = tqdm(total=evaluate_steps)
            pbar.set_description('Evaluate Processing:')
        for step in range(0, evaluate_steps):
            hiddens = []  # driver端
            for device, worker in self._workers.items():
                hidden = worker.base_forward(stage="eval", compress=compress)
                hiddens.append(hidden.to(self.device_y))
            if verbose > 0:
                pbar.update(1)
            metrics = self._workers[self.device_y].evaluate(*hiddens, compress=compress)
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
        save_traces=True,
    ):
        """Vertical split learning save model interface

        Args:
            base_model_path: base model path,only support format like 'a/b/c', where c is the model name
            fuse_model_path: fuse model path
            is_test: whether is test mode
            save_traces: (only applies to SavedModel format) When enabled,
                the SavedModel will store the function traces for each layer.
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
            assert not base_model_path[device].endswith(
                "/"
            ), f"model path should be 'a/b/c' not 'a/b/c/'"
            base_model_dir, base_model_name = base_model_path[device].rsplit("/", 1)

            if is_test:
                # only execute when unittest
                base_model_dir = os.path.join(
                    base_model_dir, device.__str__().strip("_")
                )
            res.append(
                worker.save_base_model(
                    os.path.join(base_model_dir, base_model_name),
                    save_traces=save_traces,
                )
            )

        res.append(
            self._workers[self.device_y].save_fuse_model(
                fuse_model_path, save_traces=save_traces
            )
        )
        wait(res)

    def load_model(
        self,
        base_model_path: Union[str, Dict[PYU, str]] = None,
        fuse_model_path: str = None,
        is_test=False,
        base_custom_objects=None,
        fuse_custom_objects=None,
    ):
        """Vertical split learning load model interface

        Args:
            base_model_path: base model path
            fuse_model_path: fuse model path
            is_test: whether is test mode
            base_custom_objects: Optional dictionary mapping names (strings) to custom
                classes or functions of the base model to be considered during deserialization
            fuse_custom_objects: Optional dictionary mapping names (strings) to custom
                classes or functions of the base model to be considered during deserialization.
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
            assert not base_model_path[device].endswith(
                "/"
            ), f"model path should be 'a/b/c' not 'a/b/c/'"
            base_model_dir, base_model_name = base_model_path[device].rsplit("/", 1)

            if is_test:
                # only execute when unittest
                base_model_dir = os.path.join(
                    base_model_dir, device.__str__().strip("_")
                )
            res.append(
                worker.load_base_model(
                    os.path.join(base_model_dir, base_model_name),
                    custom_objects=base_custom_objects,
                )
            )

        res.append(
            self._workers[self.device_y].load_fuse_model(
                fuse_model_path, custom_objects=fuse_custom_objects
            )
        )
        wait(res)
