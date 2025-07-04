# Copyright 2024 Ant Group Co., Ltd.
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


"""FedModel"""
import copy
import logging
import math
import os
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from ray import logger

from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray
from secretflow.device import PYU, reveal, wait
from secretflow.device.device.pyu import PYUObject
from secretflow.utils.random import global_random
from secretflow_fl.ml.nn import FLModel
from secretflow_fl.ml.nn.callbacks.callbacklist import CallbackList
from secretflow_fl.ml.nn.fl.compress import COMPRESS_STRATEGY, do_compress
from secretflow_fl.ml.nn.fl.strategy_dispatcher import dispatch_strategy
from secretflow_fl.ml.nn.metrics import Metric, aggregate_metrics
from secretflow_fl.utils.compressor import sparse_encode

from .backdoor_fl_torch import poison_dataset


class FLModel_bd(FLModel):
    def __init__(
        self,
        server=None,
        device_list: List[PYU] = [],
        model: Union["TorchModel", Callable[[], "tensorflow.keras.Model"]] = None,  # type: ignore
        aggregator=None,
        strategy="fed_avg_w",
        consensus_num=1,
        backend="tensorflow",
        random_seed=None,
        skip_bn=False,
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
            skip_bn: Whether to skip batch normalization layers when aggregate models
        """
        super().__init__(
            server,
            device_list,
            model,
            aggregator,
            strategy,
            consensus_num,
            backend,
            random_seed,
            skip_bn,
            **kwargs,
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
        sampler_method="batch",
        random_seed=None,
        dp_spent_step_freq=None,
        audit_log_dir=None,
        dataset_builder: Dict[PYU, Callable] = None,
        wait_steps=100,
        attack_party=None,
        attack_eta=1.0,
        attack_epoch=50,
    ) -> Dict:
        """Horizontal federated training interface for backdoor attack.

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
            attack_party: The party to perform backdoor attack
            attack_eta: \eta in paper How To Backdoor Federated Learning(https://arxiv.org/pdf/1807.00459)
            attack_epoch: The epoch to begin epoch
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
            ), "dp_spent_step_freq should be a integer and greater than or equal to 1!"

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

                    callbacks.on_train_batch_inner_before(epoch, device)

                    client_params, sample_num = self._workers[device].train_step(
                        client_params,
                        epoch * train_steps_per_epoch + step,
                        (
                            aggregate_freq
                            if step + aggregate_freq < train_steps_per_epoch
                            else train_steps_per_epoch - step
                        ),
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
                        self.kwargs.get("sparsity", 0.0),
                        server_weight,
                        agg_update,
                        self._res,
                    )
                    # Do sparse matrix encoding
                    if self.strategy == "fed_stc":
                        model_params = model_params.to(self.server)
                        model_params = self.server(sparse_encode, num_return=1)(
                            data=model_params,
                            encode_method="coo",
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
                        logging.debug(f"DP privacy accountant {privacy_spent}")
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

    def evaluate_bd(
        self,
        x: Union[HDataFrame, FedNdarray, Dict],
        y: Union[HDataFrame, FedNdarray, str] = None,
        batch_size: Union[int, Dict[PYU, int]] = 32,
        sample_weight: Union[HDataFrame, FedNdarray] = None,
        label_decoder=None,
        return_dict=False,
        sampler_method="batch",
        random_seed=None,
        dataset_builder: Dict[PYU, Callable] = None,
        attack_party=None,
        target_label=None,
    ) -> Tuple[
        Union[List[Metric], Dict[str, Metric]],
        Union[Dict[str, List[Metric]], Dict[str, Dict[str, Metric]]],
    ]:
        """Horizontal federated offline backdoor evaluation interface

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
            attack_party: The party to perform backdoor attack
            target_label: Attacker's target label
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

        def init_poison_val_dataset(worker, poison_rate, target_label):
            worker.benign_eval_set = copy.deepcopy(worker.eval_set)
            worker.eval_set = poison_dataset(worker.eval_set, poison_rate, target_label)

        local_metrics = {}
        metric_objs = {}
        for device, worker in self._workers.items():
            if device == attack_party:
                worker.apply(init_poison_val_dataset, 1.0, target_label)
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
