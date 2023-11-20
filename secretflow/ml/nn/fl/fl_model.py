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
import logging
import math
import os
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray
from secretflow.device import PYU, reveal, wait
from secretflow.device.device.pyu import PYUObject
from secretflow.ml.nn.fl.compress import COMPRESS_STRATEGY, do_compress
from secretflow.ml.nn.fl.strategy_dispatcher import dispatch_strategy
from secretflow.ml.nn.metrics import Metric, aggregate_metrics
from secretflow.utils.compressor import sparse_encode
from secretflow.utils.random import global_random
from secretflow.ml.nn.callbacks.callbacklist import CallbackList


class FLModel:
    def __init__(
        self,
        server=None,
        device_list: List[PYU] = [],
        model: Union['TorchModel', Callable[[], 'tensorflow.keras.Model']] = None,
        aggregator=None,
        strategy='fed_avg_w',
        consensus_num=1,
        backend="tensorflow",
        random_seed=None,
        **kwargs,  # other parameters specific to strategies
    ):
        """Interface for horizontal federated learning
        Attributes:
            server: PYU, Which PYU as a server
            device_list: party list
            model: model definition function
            aggregator:  Security aggregators can be selected according to the security level, server will do aggregate if aggregator is None
            strategy: Federated training strategy
            consensus_num: Num parties of consensus,Some strategies require multiple parties to reach consensus,
            backend: Engine backend, the backend needs to be consistent with the model type
            random_seed: If specified, the initial value of the model will remain the same, which ensures reproducible
            server_agg_method: If aggregator is none, server will use server_agg_method to aggregate params, The server_agg_method should be a function
                that takes in a list of parameter values from different parties and returns the aggregated parameter value list
        """
        if backend == "tensorflow":
            import secretflow.ml.nn.fl.backend.tensorflow.strategy  # noqa
        elif backend == "torch":
            import secretflow.ml.nn.fl.backend.torch.strategy  # noqa
        else:
            raise Exception(f"Invalid backend = {backend}")
        self.num_gpus = kwargs.get('num_gpus', 0)
        self.init_workers(
            model,
            device_list=device_list,
            strategy=strategy,
            backend=backend,
            random_seed=random_seed,
            num_gpus=self.num_gpus,
        )
        self.server = server
        self.device_list = device_list
        self._aggregator = aggregator
        self.consensus_num = consensus_num
        self.kwargs = kwargs
        self.strategy = strategy
        self._res: List[np.ndarray] = []
        self.backend = backend
        self.dp_strategy = kwargs.get('dp_strategy', None)
        self.simulation = kwargs.get('simulation', False)
        self.server_agg_method = kwargs.get('server_agg_method', None)

    def init_workers(
        self,
        model,
        device_list,
        strategy,
        backend,
        random_seed,
        num_gpus,
    ):
        self._workers = {
            device: dispatch_strategy(
                strategy,
                backend,
                builder_base=model,
                device=device,
                random_seed=random_seed,
                num_gpus=num_gpus,
            )
            for device in device_list
        }

    def initialize_weights(self):
        clients_weights = []
        initial_weight = None
        for device, worker in self._workers.items():
            weights = worker.get_weights()
            clients_weights.append(weights)
        if self._aggregator is not None:
            initial_weight = self._aggregator.average(clients_weights, axis=0)
            for device, worker in self._workers.items():
                weights = (
                    initial_weight.to(device) if initial_weight is not None else None
                )
                worker.set_weights(weights)
        else:
            clients_weights = [weight.to(self.server) for weight in clients_weights]

            if self.server_agg_method is not None:
                initial_weight = self.server(
                    self.server_agg_method,
                    num_returns=len(
                        self.device_list,
                    ),
                )(clients_weights)

                for idx, device in enumerate(self._workers.keys()):
                    weights = (
                        initial_weight[idx].to(device) if initial_weight[idx] else None
                    )
                    self._workers[device].set_weights(weights)
            else:
                raise Exception(f"aggregator and server_agg_method cannot both be none")

        return initial_weight

    def _handle_file(
        self,
        train_dict: Dict[PYU, str],
        label: str = None,
        batch_size: Union[int, Dict[PYU, int]] = 32,
        sampling_rate=None,
        shuffle=False,
        random_seed=1234,
        epochs=1,
        stage="train",
        label_decoder=None,
        max_batch_size=20000,
        prefetch_buffer_size=None,
        dataset_builder=None,
    ):
        if dataset_builder:
            steps_per_epochs = []
            for device, worker in self._workers.items():
                assert (
                    device in dataset_builder
                ), f"party={device} does not provide dataset_builder, please check"
                steps_per_epoch = worker.build_dataset_from_builder(
                    dataset_builder[device],
                    train_dict[device],
                    label,
                    None,
                    stage=stage,
                    repeat_count=epochs,
                )
                steps_per_epochs.append(steps_per_epoch)
            set_steps_per_epochs = set(reveal(steps_per_epochs))
            assert (
                len(set_steps_per_epochs) == 1
            ), "steps_per_epochs of all parties must be same"
            steps_per_epoch = set_steps_per_epochs.pop()

        else:
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
                    logging.warning(
                        "Batchsize is too large it will be set to the data size"
                    )
            # check batchsize
            for length in parties_length.values():
                batch_size = math.floor(length * sampling_rate)
                assert (
                    batch_size < max_batch_size
                ), f"Automatic batchsize is too big(batch_size={batch_size}), variable batchsize in dict is recommended"
            assert (
                sampling_rate <= 1.0 and sampling_rate > 0.0
            ), f'invalid sampling rate {sampling_rate}'
            steps_per_epoch = math.ceil(1.0 / sampling_rate)

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
        return steps_per_epoch

    def _handle_data(
        self,
        train_x: Union[HDataFrame, FedNdarray],
        train_y: Union[HDataFrame, FedNdarray] = None,
        batch_size: Union[int, Dict[PYU, int]] = 32,
        sampling_rate=None,
        shuffle=False,
        random_seed=1234,
        epochs=1,
        sample_weight: Union[FedNdarray, HDataFrame] = None,
        dataset_builder: Callable = None,
        sampler_method="batch",
        stage="train",
    ):
        assert isinstance(
            batch_size, (int, dict)
        ), f'Batch size shall be int or dict but got {type(batch_size)}.'
        if train_x is not None and train_y is not None:
            assert type(train_x) == type(
                train_y
            ), "train_x and train_y must be same data type"

        if isinstance(train_x, HDataFrame):
            train_x = train_x.values
        if isinstance(train_y, HDataFrame):
            train_y = train_y.values
        if isinstance(sample_weight, HDataFrame):
            sample_weight = sample_weight.values

        parties_length = {
            device: shape[0] for device, shape in train_x.partition_shape().items()
        }
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
            logging.warning("Batch size is too large it will be set to the data size")
        # check batch size
        for length in parties_length.values():
            batch_size = math.floor(length * sampling_rate)
            assert (
                batch_size < 1024
            ), f"Automatic batch size is too big(batch_size={batch_size}), variable batch size in dict is recommended"
        assert sampling_rate <= 1.0 and sampling_rate > 0.0, 'invalid sampling rate'
        steps_per_epoch = math.ceil(1.0 / sampling_rate)

        for device, worker in self._workers.items():
            repeat_count = epochs
            if sample_weight is not None:
                sample_weight_partitions = sample_weight.partitions[device]
            else:
                sample_weight_partitions = None
            if train_y is not None:
                y_partitions = train_y.partitions[device]
            else:
                y_partitions = None

            if dataset_builder:
                assert (
                    device in dataset_builder
                ), f"party={device} does not provide dataset_builder, please check"

                worker.build_dataset_from_builder(
                    dataset_builder[device],
                    train_x.partitions[device],
                    y_partitions,
                    s_w=sample_weight_partitions,
                    repeat_count=repeat_count,
                    stage=stage,
                )
            else:
                worker.build_dataset(
                    train_x.partitions[device],
                    y_partitions,
                    s_w=sample_weight_partitions,
                    sampling_rate=sampling_rate,
                    shuffle=shuffle,
                    random_seed=random_seed,
                    repeat_count=repeat_count,
                    sampler_method=sampler_method,
                    stage=stage,
                )
        return steps_per_epoch

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
        sampler_method='batch',
        random_seed=None,
        dp_spent_step_freq=None,
        audit_log_dir=None,
        dataset_builder: Dict[PYU, Callable] = None,
        wait_steps=100,
    ) -> Dict:
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
            sampler_method: The name of sampler method
            random_seed: Prg seed for shuffling
            dp_spent_step_freq: specifies how many training steps to check the budget of dp
            audit_log_dir: path of audit log dir, checkpoint will be save if audit_log_dir is not None
            dataset_builder: Callable function about hot to build the dataset. must return (dataset, steps_per_epoch)
            wait_steps: A step size to indicate how many concurrent tasks should be waited, which could prevent the stuck of ray when more tasks join (default 100).
        Returns:
            A history object. It's history.global_history attribute is a
            aggregated record of training loss values and metrics, while
            history.local_history attribute is a record of training loss
            values and metrics of each party.
        """
        if not random_seed:
            random_seed = global_random([*self._workers][0], 100000)

        params = locals()
        logging.info(f"FL Train Params: {params}")

        # sanity check
        if self._aggregator is None:
            if self.server_agg_method is None or self.server is None:
                raise Exception(
                    "When aggregator is none, neither the server nor the server_agg_method can be none"
                )
        assert isinstance(validation_freq, int) and validation_freq >= 1
        assert isinstance(aggregate_freq, int) and aggregate_freq >= 1
        if dp_spent_step_freq is not None:
            assert (
                isinstance(dp_spent_step_freq, int) and dp_spent_step_freq >= 1
            ), 'dp_spent_step_freq should be a integer and greater than or equal to 1!'

        # build dataset
        if isinstance(x, Dict):
            if validation_data is not None:
                valid_x, valid_y = validation_data, y
            else:
                valid_x, valid_y = None, None

            train_steps_per_epoch = self._handle_file(
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
                dataset_builder=dataset_builder,
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

            train_steps_per_epoch = self._handle_data(
                train_x,
                train_y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                sampling_rate=batch_sampling_rate,
                shuffle=shuffle,
                random_seed=random_seed,
                epochs=epochs,
                sampler_method=sampler_method,
                dataset_builder=dataset_builder,
            )
        # setup callback list
        callbacks = CallbackList(
            callbacks=callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            workers=self._workers,
            device_y=None,
            epochs=epochs,
            verbose=verbose,
            steps=train_steps_per_epoch,
        )

        initial_weight = self.initialize_weights()
        logging.debug(f"initial_weight: {initial_weight}")
        server_weight = None
        if self.server and isinstance(initial_weight, PYUObject):
            server_weight = initial_weight
        callbacks.on_train_begin()
        model_params = None
        model_params_list = None
        for epoch in range(epochs):
            res = []
            report_list = []
            # do train
            report_list.append(f"epoch: {epoch+1}/{epochs} - ")
            callbacks.on_epoch_begin(epoch=epoch)
            for step in range(0, train_steps_per_epoch, aggregate_freq):
                callbacks.on_train_batch_begin(batch=step)
                client_param_list, sample_num_list = [], []
                for idx, device in enumerate(self._workers.keys()):
                    client_params = (
                        model_params_list[idx].to(device)
                        if model_params_list is not None
                        else None
                    )
                    # refresh data-iter
                    if step == 0:
                        self.kwargs["refresh_data"] = True
                    else:
                        self.kwargs["refresh_data"] = False

                    client_params, sample_num = self._workers[device].train_step(
                        client_params,
                        epoch * train_steps_per_epoch + step,
                        aggregate_freq
                        if step + aggregate_freq < train_steps_per_epoch
                        else train_steps_per_epoch - step,
                        **self.kwargs,
                    )
                    client_param_list.append(client_params)
                    sample_num_list.append(sample_num)
                    res.append(client_params)
                if self._aggregator is not None:
                    model_params = self._aggregator.average(
                        client_param_list, axis=0, weights=sample_num_list
                    )
                else:
                    if self.server is not None:
                        # server will do aggregation
                        model_params_list = [
                            param.to(self.server) for param in client_param_list
                        ]
                        model_params_list = self.server(
                            self.server_agg_method,
                            num_returns=len(
                                self.device_list,
                            ),
                        )(model_params_list)
                        model_params_list = [
                            params.to(device)
                            for device, params in zip(
                                self.device_list, model_params_list
                            )
                        ]
                    else:
                        raise Exception(
                            "Aggregation can be on either an aggregator or a server, but not none at the same time"
                        )

                # Do weight sparsify
                if self.strategy in COMPRESS_STRATEGY and server_weight:
                    if self._res:
                        self._res.to(self.server)
                    agg_update = model_params.to(self.server)
                    server_weight = server_weight.to(self.server)
                    server_weight, model_params, self._res = self.server(
                        do_compress, num_returns=3
                    )(
                        self.strategy,
                        self.kwargs.get('sparsity', 0.0),
                        server_weight,
                        agg_update,
                        self._res,
                    )
                    # Do sparse matrix encoding
                    if self.strategy == 'fed_stc':
                        model_params = model_params.to(self.server)
                        model_params = self.server(sparse_encode, num_return=1)(
                            data=model_params,
                            encode_method='coo',
                        )

                # DP operation
                if dp_spent_step_freq is not None and self.dp_strategy is not None:
                    current_dp_step = math.ceil(
                        epoch * train_steps_per_epoch / aggregate_freq
                    ) + int(step / aggregate_freq)
                    if current_dp_step % dp_spent_step_freq == 0:
                        privacy_spent = self.dp_strategy.get_privacy_spent(
                            current_dp_step
                        )
                        logging.debug(f'DP privacy accountant {privacy_spent}')
                if len(res) == wait_steps:
                    wait(res)
                    res = []
                if self._aggregator is not None:
                    model_params_list = [model_params for _ in self.device_list]
                    model_params_list = [
                        params.to(device)
                        for device, params in zip(self.device_list, model_params_list)
                    ]
                callbacks.on_train_batch_end(batch=step)

            # last batch
            for idx, device in enumerate(self._workers.keys()):
                client_params = model_params_list[idx].to(device)
                self._workers[device].apply_weights(client_params)
            model_params_list = None

            local_metrics_obj = []
            for device, worker in self._workers.items():
                local_metrics_obj.append(worker.wrap_local_metrics())

            if epoch % validation_freq == 0 and valid_x is not None:
                callbacks.on_test_begin()
                global_eval, local_eval = self.evaluate(
                    valid_x,
                    valid_y,
                    batch_size=batch_size,
                    sample_weight=sample_weight,
                    return_dict=True,
                    label_decoder=label_decoder,
                    random_seed=random_seed,
                    sampler_method=sampler_method,
                    dataset_builder=dataset_builder,
                )
                for device, worker in self._workers.items():
                    worker.set_validation_metrics(global_eval)

                # save checkpoint
                if audit_log_dir is not None:
                    epoch_model_path = os.path.join(
                        audit_log_dir, "base_model", str(epoch)
                    )
                    self.save_model(
                        model_path=epoch_model_path, is_test=self.simulation
                    )
                callbacks.on_test_end()

            stop_trainings = [
                reveal(worker.get_stop_training()) for worker in self._workers.values()
            ]
            if sum(stop_trainings) >= self.consensus_num:
                break
            callbacks.on_epoch_end(epoch=epoch)
        callbacks.on_train_end()
        return callbacks.history

    def predict(
        self,
        x: Union[HDataFrame, FedNdarray, Dict],
        batch_size=None,
        label_decoder=None,
        sampler_method='batch',
        random_seed=1234,
        dataset_builder: Dict[PYU, Callable] = None,
    ) -> Dict[PYU, PYUObject]:
        """Horizontal federated offline prediction interface

        Args:
            x: feature, FedNdArray or HDataFrame
            batch_size: Number of samples per gradient update, int or Dict
            label_decoder: Only used for CSV reading, for label preprocess
            sampler_method: The name of sampler method
            random_seed: Prg seed for shuffling
            dataset_builder: Callable function about hot to build the dataset. must return (dataset, steps_per_epoch)
        Returns:
            predict results, numpy.array
        """
        if not random_seed:
            random_seed = global_random([*self._workers][0], 100000)

        if isinstance(x, Dict):
            predict_steps = self._handle_file(
                x,
                None,
                batch_size=batch_size,
                stage="eval",
                epochs=1,
                label_decoder=label_decoder,
                dataset_builder=dataset_builder,
            )
        else:
            if isinstance(x, HDataFrame):
                eval_x = x.values
            else:
                eval_x = x

            predict_steps = self._handle_data(
                eval_x,
                train_y=None,
                sample_weight=None,
                batch_size=batch_size,
                stage="eval",
                epochs=1,
                sampler_method=sampler_method,
                random_seed=random_seed,
                dataset_builder=dataset_builder,
            )

        result = {}

        for device, worker in self._workers.items():
            pred = worker.predict(predict_steps)
            result[device] = pred

        return result

    def evaluate(
        self,
        x: Union[HDataFrame, FedNdarray, Dict],
        y: Union[HDataFrame, FedNdarray, str] = None,
        batch_size: Union[int, Dict[PYU, int]] = 32,
        sample_weight: Union[HDataFrame, FedNdarray] = None,
        label_decoder=None,
        return_dict=False,
        sampler_method='batch',
        random_seed=None,
        dataset_builder: Dict[PYU, Callable] = None,
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
            return_dict: If `True`, loss and metric results are returned as a dict,
                with each key being the name of the metric. If `False`, they are
                returned as a list.
            sampler_method: The name of sampler method.
            dataset_builder: Callable function about hot to build the dataset. must return (dataset, steps_per_epoch)

        Returns:
            A tuple of two objects. The first object is a aggregated record of
            metrics, and the second object is a record of training loss values
            and metrics of each party.
        """
        if not random_seed:
            random_seed = global_random([*self._workers][0], 100000)
        if isinstance(x, Dict):
            evaluate_steps = self._handle_file(
                x,
                y,
                batch_size=batch_size,
                stage="eval",
                epochs=1,
                label_decoder=label_decoder,
                dataset_builder=dataset_builder,
            )
        else:
            assert type(x) == type(y), "x and y must be same data type"
            if isinstance(x, HDataFrame) and isinstance(y, HDataFrame):
                eval_x, eval_y = x.values, y.values
            else:
                eval_x, eval_y = x, y
            if isinstance(sample_weight, HDataFrame):
                sample_weight = sample_weight.values

            evaluate_steps = self._handle_data(
                eval_x,
                eval_y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                stage="eval",
                epochs=1,
                sampler_method=sampler_method,
                random_seed=random_seed,
                dataset_builder=dataset_builder,
            )

        local_metrics = {}
        metric_objs = {}
        for device, worker in self._workers.items():
            metric_objs[device.party] = worker.evaluate(evaluate_steps)
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

    def save_model(
        self,
        model_path: Union[str, Dict[PYU, str]],
        is_test=False,
        saved_model=False,
    ):
        """Horizontal federated save model interface

        Args:
            model_path: model path, only support format like 'a/b/c', where c is the model name
            is_test: whether is test mode
            saved_model: bool Whether to save as savedmodel or torchscript format
        """
        assert isinstance(
            model_path, (str, Dict)
        ), f'Model path accepts string or dict but got {type(model_path)}.'

        if isinstance(model_path, str):
            model_path = {device: model_path for device in self._workers.keys()}

        res = []
        for device, worker in self._workers.items():
            assert device in model_path, f'Should provide a path for device {device}.'
            assert not model_path[device].endswith(
                "/"
            ), f"model path should be 'a/b/c' not 'a/b/c/'"
            device_model_path, device_model_name = model_path[device].rsplit("/", 1)
            if is_test:
                device_model_path = os.path.join(
                    device_model_path, device.__str__().strip("_")
                )
            if saved_model:
                raise Exception(
                    "Not implement yet, it will be implemented in subsequent versions"
                )
            else:
                res.append(
                    worker.save_model(
                        os.path.join(device_model_path, device_model_name)
                    )
                )
        wait(res)

    def load_model(
        self,
        model_path: Union[str, Dict[PYU, str]],
        is_test=False,
        saved_model=False,
        force_all_participate=False,
    ):
        """Horizontal federated load model interface

        Args:
            model_path: model path
            is_test: whether is test mode
            saved_model: bool Whether to load from savedmodel or torchscript format
        """
        assert isinstance(
            model_path, (str, Dict)
        ), f'Model path accepts string or dict but got {type(model_path)}.'
        if isinstance(model_path, str):
            model_path = {device: model_path for device in self._workers.keys()}

        res = []
        for device, worker in self._workers.items():
            assert device in model_path, f'Should provide a path for device {device}.'
            device_model_path, device_model_name = model_path[device].rsplit("/", 1)

            if is_test:
                device_model_path = os.path.join(
                    device_model_path, device.__str__().strip("_")
                )
            if saved_model:
                raise Exception(
                    "Not implement yet, it will be implemented in subsequent versions"
                )
            else:
                res.append(
                    worker.load_model(
                        os.path.join(device_model_path, device_model_name)
                    )
                )
        checks = reveal(res)
        if force_all_participate:
            assert len(set(checks)) == 1, "return of all parties must be same"
            logging.info(f"load model success")
        else:
            logging.info(f"load model success")
