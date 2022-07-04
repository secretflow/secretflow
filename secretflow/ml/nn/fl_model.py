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


"""FedModel

"""
import math
import os
import secrets
import logging
from typing import Callable, Dict, List, Tuple, Union

import tensorflow as tf
from secretflow.data.horizontal.dataframe import HDataFrame
from secretflow.data.ndarray import FedNdarray
from secretflow.device import PYU, reveal, wait
from secretflow.ml.nn.fl_base import PYUTFModel
from secretflow.ml.nn.metrics import Metric, aggregate_metrics
from secretflow.ml.nn.utils import History
from tqdm import tqdm


class FLModelTF:
    def __init__(
        self,
        device_list: List[PYU] = [],
        model: Callable[[], tf.keras.Model] = None,
        aggregator=None,
        sampler='batch',
    ):

        self._workers = {
            device: PYUTFModel(model, device=device) for device in device_list
        }
        self._aggregator = aggregator
        self._sampler = sampler
        self.steps_per_epoch = 0
        self.consensus_num = 1

    def handle_file(
        self,
        train_dict: Dict[PYU, str],
        label: str,
        batch_size: Union[int, Dict[PYU, int]] = 32,
        sampling_rate=None,
        shuffle=False,
        random_seed=1234,
        epochs=1,
        stage="train",
        label_decoder=None,
        max_batch_size=20000,
        prefetch_buffer_size=None,
    ):
        # get party length
        parties_length = reveal(
            {
                device: worker.get_rows_count(train_dict[device])
                for device, worker in self._workers.items()
            }
        )
        if sampling_rate is None:
            if isinstance(batch_size, int):
                sampling_rate = max(
                    [batch_size / length for length in parties_length.values()]
                )
            else:
                sampling_rate = max(
                    [
                        batch_size[device] / length
                        for device, length in parties_length.items()
                    ]
                )
            if sampling_rate > 1.0:
                sampling_rate = 1.0
        # check batchsize
        for length in parties_length.values():
            batch_size = math.floor(length * sampling_rate)
            assert (
                batch_size < max_batch_size
            ), f"Automatic batchsize is too big(batch_size={batch_size}), variable batchsize in dict is recommended"
        assert sampling_rate <= 1.0 and sampling_rate > 0.0, 'invalid sampling rate'
        self.steps_per_epoch = math.ceil(1.0 / sampling_rate)

        for device, worker in self._workers.items():
            repeat_count = epochs
            worker.build_dataset_from_csv(
                train_dict[device],
                label,
                sampling_rate=sampling_rate,
                shuffle=shuffle,
                random_seed=random_seed,
                repeat_count=repeat_count,
                sample_length=parties_length[device],
                prefetch_buffer_size=prefetch_buffer_size,
                stage=stage,
                label_decoder=label_decoder,
            )

    def handle_data(
        self,
        train_x: Union[HDataFrame, FedNdarray],
        train_y: Union[HDataFrame, FedNdarray],
        batch_size: Union[int, Dict[PYU, int]] = 32,
        sampling_rate=None,
        shuffle=False,
        random_seed=1234,
        epochs=1,
        sample_weight: Union[FedNdarray, HDataFrame] = None,
    ):
        assert isinstance(
            batch_size, (int, dict)
        ), f'Batch size shall be int or dict but got {type(batch_size)}.'
        assert type(train_x) == type(
            train_y
        ), "train_x and train_y must be same data type"

        if isinstance(train_x, HDataFrame):
            train_x, train_y = train_x.values, train_y.values

        if isinstance(sample_weight, HDataFrame):
            sample_weight = sample_weight.values
        parties_length = train_x.length()
        if sampling_rate is None:
            if isinstance(batch_size, int):
                sampling_rate = max(
                    [batch_size / length for length in parties_length.values()]
                )
            else:
                sampling_rate = max(
                    [
                        batch_size[device] / length
                        for device, length in parties_length.items()
                    ]
                )
        # check batchsize
        for length in parties_length.values():
            batch_size = math.floor(length * sampling_rate)
            assert (
                batch_size < 1024
            ), f"Automatic batchsize is too big(batch_size={batch_size}), variable batchsize in dict is recommended"
        assert sampling_rate <= 1.0 and sampling_rate > 0.0, 'invalid sampling rate'
        self.steps_per_epoch = math.ceil(1.0 / sampling_rate)

        for device, worker in self._workers.items():
            repeat_count = epochs
            if sample_weight is not None:
                sample_weight_partition = sample_weight.partitions[device]
            else:
                sample_weight_partition = None
            worker.build_dataset(
                train_x.partitions[device],
                train_y.partitions[device],
                s_w=sample_weight_partition,
                sampling_rate=sampling_rate,
                shuffle=shuffle,
                random_seed=random_seed,
                repeat_count=repeat_count,
                sampler=self._sampler,
            )

    def fit(
        self,
        x: Union[HDataFrame, FedNdarray, Dict[PYU, str]],
        y: Union[HDataFrame, FedNdarray, str],
        batch_size: Union[int, Dict[PYU, int]] = 32,
        batch_sampling_rate: float = None,
        epochs: int = 1,
        verbose: int = 1,
        callbacks=None,
        validation_data=None,
        shuffle=False,
        class_weight=None,
        sample_weight=None,
        validation_freq=1,
        aggregate_freq=1,
        label_decoder=None,
        max_batch_size=20000,
        prefetch_buffer_size=None,
    ) -> History:
        """Horizontal federated training interface

        Args:
            x: feature, FedNdArray, HDataFrame or Dict {PYU: model_path}
            y: label, FedNdArray, HDataFrame or str(column name of label)
            batch_size: Number of samples per gradient update, int or Dict, recommend 64 or more for safety
            batch_sampling_rate: Ratio of sample per batch, float
            epochs: Number of epochs to train the model
            verbose: 0, 1. Verbosity mode
            callbacks: List of `keras.callbacks.Callback` instances.
            validation_data: Data on which to evaluate
            shuffle: whether to shuffle the training data
            class_weight: Dict mapping class indices (integers) to a weight (float)
            sample_weight: weights for the training samples
            validation_freq: specifies how many training epochs to run before a new validation run is performed
            aggregate_freq: Number of steps of aggregation
            label_decoder: Only used for CSV reading, for label preprocess
            max_batch_size: Max limit of batch size
            prefetch_buffer_size: An int specifying the number of feature batches to prefetch for performance improvement. Only for csv reader
        Returns:
            A history object. It's history.global_history attribute is a
            aggregated record of trainning loss values and metrics, while
            history.local_history attribute is a record of trainning loss
            values and metrics of each party.
        """
        random_seed = secrets.randbelow(100000)
        logging.info(
            f"FL Train Params: batch_size={batch_size} epochs={epochs} shuffle={shuffle} random_seed={random_seed} validation_freq={validation_freq} aggregate_freq={aggregate_freq}"
        )

        # sanity check
        assert isinstance(validation_freq, int) and validation_freq >= 1
        assert isinstance(aggregate_freq, int) and aggregate_freq >= 1
        # build dataset

        if isinstance(x, Dict):
            if validation_data is not None:
                valid_x, valid_y = validation_data, y
            else:
                valid_x, valid_y = None, None

            self.handle_file(
                x,
                y,
                batch_size=batch_size,
                sampling_rate=batch_sampling_rate,
                shuffle=shuffle,
                random_seed=random_seed,
                epochs=epochs,
                label_decoder=label_decoder,
                max_batch_size=max_batch_size,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        else:
            assert type(x) == type(y), "x and y must be same data type"
            if isinstance(x, HDataFrame) and isinstance(y, HDataFrame):
                train_x, train_y = x.values, y.values
            else:
                train_x, train_y = x, y

            if validation_data is not None:
                valid_x, valid_y = validation_data[0], validation_data[1]
            else:
                valid_x, valid_y = None, None

            self.handle_data(
                train_x,
                train_y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                sampling_rate=batch_sampling_rate,
                shuffle=shuffle,
                random_seed=random_seed,
                epochs=epochs,
            )
        history = History()

        current_weight = None

        for device, worker in self._workers.items():
            worker.init_training(callbacks, epochs=epochs)
            worker.on_train_begin()
        for epoch in range(epochs):
            report_list = []
            pbar = tqdm(total=self.steps_per_epoch)
            # do train
            report_list.append(f"epoch: {epoch}/{epochs} - ")
            [worker.on_epoch_begin(epoch) for device, worker in self._workers.items()]
            for step in range(0, self.steps_per_epoch, aggregate_freq):
                if verbose == 1:
                    pbar.update(aggregate_freq)
                weights, sample_nums = [], []
                for device, worker in self._workers.items():
                    weight = (
                        current_weight.to(device)
                        if current_weight is not None
                        else None
                    )
                    weight, sample_num = worker.train_step(
                        weight,
                        epoch * self.steps_per_epoch + step,
                        aggregate_freq
                        if step + aggregate_freq < self.steps_per_epoch
                        else self.steps_per_epoch - step,
                    )
                    weights.append(weight)
                    sample_nums.append(sample_num)

                current_weight = self._aggregator.average(
                    weights, axis=0, weights=sample_nums
                )

            local_metrics_obj = []
            for device, worker in self._workers.items():
                worker.on_epoch_end(epoch)
                local_metrics_obj.append(worker.wrap_local_metrics())

            local_metrics = reveal(local_metrics_obj)
            for local_metric in local_metrics:
                history.record_local_history(party=device.party, metrics=local_metric)

            g_metrics = aggregate_metrics(local_metrics)
            history.record_global_history(g_metrics)

            if epoch % validation_freq == 0 and valid_x is not None:
                global_eval, local_eval = self.evaluate(
                    valid_x,
                    valid_y,
                    batch_size=batch_size,
                    verbose=0,
                    sample_weight=sample_weight,
                    return_dict=True,
                    label_decoder=label_decoder,
                )
                for device, worker in self._workers.items():
                    worker.set_validation_metrics(global_eval)
                    history.record_local_history(
                        party=device.party,
                        metrics=local_eval[device.party].values(),
                        stage="val",
                    )
                history.record_global_history(metrics=global_eval.values(), stage="val")

            for name, metric in history.global_history.items():
                report_list.append(f"{name}:{metric[-1]} ")

            report = " ".join(report_list)
            if verbose == 1:
                pbar.set_postfix_str(report)
                pbar.close()
            stop_trainings = [
                reveal(worker.get_stop_training()) for worker in self._workers.values()
            ]
            if sum(stop_trainings) >= self.consensus_num:
                break

        return history

    def predict(
        self,
        x: Union[HDataFrame, FedNdarray],
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ) -> List:
        """Horizontal federated offline prediction interface

        Args:
            x: feature, FedNdArray or HDataFrame
            batch_size: Number of samples per gradient update, int or Dict
            verbose: 0, 1. Verbosity mode
            steps: Total number of steps (batches of samples)
            callbacks: List of `keras.callbacks.Callback` instances.
            max_queue_size: Used for generator or `keras.utils.Sequence` input only
            workers: Used for generator or `keras.utils.Sequence` input only.
            use_multiprocessing: Whether use multi process to predict
        Returns:
            predict results, numpy.array
        """
        parties_length = x.length()
        parties_batch_size = {}
        if batch_size is None:
            parties_batch_size = dict.fromkeys(parties_length.keys())
        if isinstance(batch_size, int):
            parties_batch_size = dict.romkeys(parties_length.keys(), batch_size)
        if isinstance(batch_size, Dict):
            parties_batch_size = batch_size

        y_pred = [
            worker.predict(
                x.partitions[device].data,
                batch_size=parties_batch_size[device],
                verbose=verbose,
                steps=steps,
                callbacks=callbacks,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
            )
            for device, worker in self._workers.items()
        ]
        return y_pred

    def evaluate(
        self,
        x: Union[HDataFrame, FedNdarray, Dict],
        y: Union[HDataFrame, FedNdarray, str] = None,
        batch_size: Union[int, Dict[PYU, int]] = 32,
        sample_weight: Union[HDataFrame, FedNdarray] = None,
        label_decoder=None,
        verbose=0,
        steps=None,
        return_dict=False,
    ) -> Tuple[
        Union[List[Metric], Dict[str, Metric]],
        Union[Dict[str, List[Metric]], Dict[str, Dict[str, Metric]]],
    ]:
        """Horizontal federated offline evaluation interface

        Args:
            x: Input data. It could be:
                - FedNdArray
                - HDataFrame
                - Dict {PYU: model_path}
            y: Label. It could be:
                - FedNdArray
                - HDataFrame
                - str column name of csv
            batch_size: Integer or `Dict`. Number of samples per batch of
                computation. If unspecified, `batch_size` will default to 32.
            sample_weight: Optional Numpy array of weights for the test samples,
                used for weighting the loss function.
            label_decoder: User define how to handle label column when use csv reader
            verbose: Verbosity mode. 0 = silent, 1 = progress bar.
            steps: Integer or `None`. Total number of steps (batches of samples)
            return_dict: If `True`, loss and metric results are returned as a dict,
                with each key being the name of the metric. If `False`, they are
                returned as a list.

        Returns:
            A tuple of two objects. The first object is a aggregated record of
            metrics, and the second object is a record of trainning loss values
            and metrics of each party.
        """
        local_metrics = {}
        if isinstance(x, Dict):
            parties_length = reveal(
                {
                    device: worker.get_rows_count(x[device])
                    for device, worker in self._workers.items()
                }
            )
            parties_batch_size = {}
            if isinstance(batch_size, int):
                parties_batch_size = dict.fromkeys(parties_length.keys(), batch_size)
            elif isinstance(batch_size, Dict):
                parties_batch_size = batch_size
            else:
                raise Exception("Illegal batch_size")

            self.handle_file(
                x,
                y,
                batch_size=batch_size,
                stage="valid",
                label_decoder=label_decoder,
            )
            metric_objs = {}
            for device, worker in self._workers.items():
                worker.evaluate(
                    None,
                    None,
                    batch_size=parties_batch_size[device],
                    verbose=verbose,
                    steps=steps,
                )
                metric_objs[device.party] = worker.wrap_local_metrics()
            local_metrics = reveal(metric_objs)
        else:
            assert type(x) == type(y), "x and y must be same data type"
            if isinstance(x, HDataFrame):
                x, y = x.values, y.values

            if isinstance(sample_weight, HDataFrame):
                sample_weight = sample_weight.values

            parties_length = x.length()
            parties_batch_size = {}
            if isinstance(batch_size, int):
                parties_batch_size = dict.fromkeys(parties_length.keys(), batch_size)
            if isinstance(batch_size, Dict):
                parties_batch_size = batch_size

            metric_objs = {}
            for device, worker in self._workers.items():
                if sample_weight is not None:
                    s_w_partition = sample_weight.partitions[device]
                else:
                    s_w_partition = None
                worker.evaluate(
                    x.partitions[device],
                    y.partitions[device],
                    batch_size=parties_batch_size[device],
                    verbose=verbose,
                    sample_weight=s_w_partition,
                    steps=steps,
                )
                metric_objs[device.party] = worker.wrap_local_metrics()
            local_metrics = reveal(metric_objs)
        g_metrics = aggregate_metrics(local_metrics.values())
        if return_dict:
            return (
                {m.name: m for m in g_metrics},
                {
                    party: {m.name: m for m in metrics}
                    for party, metrics in local_metrics.items()
                },
            )
        else:
            return g_metrics, local_metrics

    def save_model(self, model_path: Union[str, Dict[PYU, str]], is_test=False):
        """Horizontal federated save model interface

        Args:
            model_path: model path
            is_test: whether is test mode
        """
        assert isinstance(
            model_path, (str, Dict)
        ), f'Model path accepts string or dict but got {type(model_path)}.'
        if isinstance(model_path, str):
            model_path = {device: model_path for device in self._workers.keys()}

        res = []
        for device, worker in self._workers.items():
            assert device in model_path, f'Should provide a path for device {device}.'
            if is_test:
                model_path_test = os.path.join(
                    model_path[device], device.__str__().strip("_")
                )
                res.append(worker.save_model(model_path_test))
            else:
                res.append(worker.save_model(model_path[device]))
        wait(res)

    def load_model(self, model_path: Union[str, Dict[PYU, str]], is_test=False):
        """Horizontal federated load model interface

        Args:
            model_path: model path
            is_test: whether is test mode
        """
        assert isinstance(
            model_path, (str, Dict)
        ), f'Model path accepts string or dict but got {type(model_path)}.'
        if isinstance(model_path, str):
            model_path = {device: model_path for device in self._workers.keys()}

        res = []
        for device, worker in self._workers.items():
            assert device in model_path, f'Should provide a path for device {device}.'
            if is_test:
                model_path_test = os.path.join(
                    model_path[device], device.__str__().strip("_")
                )
                res.append(worker.load_model(model_path_test))
            else:
                res.append(worker.load_model(model_path[device]))
        wait(res)
