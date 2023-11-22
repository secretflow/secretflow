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
from typing import Callable, Dict, Iterable, List, Tuple, Union

from multiprocess import cpu_count

from secretflow.data import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, Device, reveal, wait
from secretflow.device.device.pyu import PYUObject
from secretflow.ml.nn.sl.agglayer.agg_layer import AggLayer
from secretflow.ml.nn.sl.agglayer.agg_method import AggMethod
from secretflow.ml.nn.sl.base import SLBaseModel
from secretflow.ml.nn.sl.strategy_dispatcher import dispatch_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.utils.random import global_random
from secretflow.ml.nn.callbacks.callbacklist import CallbackList


class SLModel:
    def __init__(
        self,
        base_model_dict: Dict[Device, Callable[[], 'tensorflow.keras.Model']] = {},
        device_y: PYU = None,
        model_fuse: Callable[[], 'tensorflow.keras.Model'] = None,
        dp_strategy_dict: Dict[Device, DPStrategy] = None,
        random_seed: int = None,
        backend: str = "tensorflow",
        strategy='split_nn',
        agg_method: AggMethod = None,
        **kwargs,
    ):
        """Interface for vertical split learning
        Attributes:
            base_model_dict: Basemodel dictionary, key is PYU, value is the Basemodel defined by party.
            device_y: Define which model have label.
            model_fuse:  Fuse model definition.
            dp_strategy_dict: Dp strategy dictionary.
            random_seed: If specified, the initial value of the model will remain the same, which ensures reproducible.
            backend: name of backend engine, tensorflow or torch, default tensorflow
            strategy: Strategy of split learning.
            agg_method: agg method decide how to aggregate hiddens from each parties, default None(compatible with legacy mode)
        Keyword Args
            simulation: Only need when use simulation mode
            base_local_steps: Only for 'split_async' strategy, Number of rounds for local base update process
            fuse_local_steps: Only for 'split_async' strategy, Number of rounds for local fuse update process
            bound_param: Only for 'split_async' strategy, Parameter for limiting local gradient change
            loss_thres: Only for 'split_state_async' strategy, Loss threshold triggering switch state in splitStateAS strategy
            split_steps: Only for 'split_state_async' strategy, Number of batches triggering switch state in splitStateAS strategy
            max_fuse_local_steps: Only for 'split_state_async' strategy, Maximum number of rounds for fuse local update in splitStateAS strategy?
            compressor: Define strategy tensor compression algorithms to speed up transmission.
            device_agg: The party do aggregation, it can be a PYU, SPU, etc.
            **kwargs: For custom strategies.
        """

        self.device_y = device_y
        self.dp_strategy_dict = dp_strategy_dict
        self.simulation = kwargs.pop('simulation', False)
        self.device_agg = kwargs.pop('device_agg', None)
        self.compressor = kwargs.pop('compressor', None)
        self.base_model_dict = base_model_dict
        self.backend = backend
        self.num_parties = len(base_model_dict)
        self.agglayer = AggLayer(
            device_agg=self.device_agg if self.device_agg else self.device_y,
            parties=list(base_model_dict.keys()),
            device_y=self.device_y,
            agg_method=agg_method,
            backend=backend,
            compressor=self.compressor,
        )
        self.pipeline_size = kwargs.get('pipeline_size', 1)
        assert self.pipeline_size >= 1, f"invalid pipeline size: {self.pipeline_size}"

        if backend.lower() == "tensorflow":
            import secretflow.ml.nn.sl.backend.tensorflow.strategy  # noqa
        elif backend.lower() == "torch":
            import secretflow.ml.nn.sl.backend.torch.strategy  # noqa
        else:
            raise Exception(f"Invalid backend = {backend}")
        worker_list = list(base_model_dict.keys())
        if device_y not in worker_list:
            worker_list.append(device_y)
        self._workers: Dict[PYU, SLBaseModel] = {}
        self.history = None
        for device in worker_list:
            self._workers[device], self.check_skip_grad = dispatch_strategy(
                strategy,
                backend=backend,
                builder_base=base_model_dict[device]
                if device in base_model_dict.keys()
                else None,
                builder_fuse=None if device != device_y else model_fuse,
                random_seed=random_seed,
                dp_strategy=dp_strategy_dict.get(device, None)
                if dp_strategy_dict
                else None,
                device=device,
                **kwargs,
            )

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
        dataset_builder: Dict = None,
    ):
        if isinstance(x, (VDataFrame, FedNdarray)):
            x = [x]
        for i in range(len(x)):
            if isinstance(x[i], (VDataFrame, HDataFrame)):
                x[i] = x[i].to_pandas()
        if isinstance(y, (VDataFrame)):
            y = y.to_pandas()
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

            if dataset_builder:
                # in dataset builder mode, xi cannot be none, or else datasetbuilder in worker cannot parse label
                xs = (
                    [
                        xi.partitions[device].data  # xi is FedDataframe
                        if isinstance(xi.partitions[device], Partition)
                        else xi.partitions[device]  # xi is FedNdarray
                        for xi in x
                    ]
                    if device in dataset_builder
                    else [None]
                )
                if device not in dataset_builder:
                    logging.warning("party={device} does not provide dataset_builder")
                    dataset_partition = None
                    if device in self.base_model_dict:
                        raise Exception(
                            "dataset builder must be supply when base_net is not none"
                        )
                else:
                    dataset_partition = dataset_builder[device]
                ret = worker.build_dataset_from_builder(
                    *xs,
                    y=y_partitions,
                    s_w=s_w_partitions,
                    batch_size=batch_size,
                    random_seed=random_seed,
                    stage=stage,
                    dataset_builder=dataset_partition,
                )
                worker_steps.append(ret)

            else:
                # if don't have feature, driver will pass None to device worker
                xs = (
                    [xi.partitions[device] for xi in x]
                    if device in self.base_model_dict
                    else [None]
                )
                xs = [t.data if isinstance(t, Partition) else t for t in xs]
                worker.build_dataset_from_numeric(
                    *xs,
                    y=y_partitions,
                    s_w=s_w_partitions,
                    batch_size=batch_size,
                    buffer_size=batch_size * 8,
                    shuffle=shuffle,
                    repeat_count=epochs,
                    stage=stage,
                    random_seed=random_seed,
                )

        parties_length = [shape[0] for shape in x[0].partition_shape().values()]
        assert len(set(parties_length)) == 1, "length of all parties must be same"
        steps_per_epoch = math.ceil(parties_length[0] / batch_size)
        if dataset_builder:
            worker_steps_per_epoch = reveal(worker_steps)
            worker_steps_per_epoch = [
                steps for steps in worker_steps_per_epoch if steps != -1
            ]
            assert (
                len(set(worker_steps_per_epoch)) == 1
            ), "steps_per_epoch of all parties must be same, Please check whether the batchsize or steps_per_epoch of all parties are consistent"
            # set worker_steps_per_epoch[0] to steps_per_epoch if databuilder return steps_per_epoch else use driver calculate result
            if worker_steps_per_epoch[0] > 0:
                steps_per_epoch = worker_steps_per_epoch[0]

        self._workers[self.device_y].set_steps_per_epoch(steps_per_epoch)
        return steps_per_epoch

    def handle_file(
        self,
        train_dict: Dict[PYU, str],
        label: str = None,
        sample_weight: Union[FedNdarray, VDataFrame] = None,
        batch_size=32,
        shuffle=False,
        epochs=1,
        stage="train",
        random_seed=1234,
        dataset_builder: Dict = None,
    ):
        worker_steps = []
        for device, worker in self._workers.items():
            s_w_partitions = (
                sample_weight.partitions[device] if sample_weight is not None else None
            )
            if dataset_builder:
                if device not in dataset_builder:
                    logging.warning("party={device} does not provide dataset_builder")
                    dataset_partition = None
                    if device in self.base_model_dict:
                        raise Exception(
                            "dataset builder must be supply when base_net is not none"
                        )
                else:
                    dataset_partition = dataset_builder[device]
                ret = worker.build_dataset_from_builder(
                    train_dict[device],
                    y=label if self.device_y == device else None,
                    s_w=s_w_partitions,
                    batch_size=batch_size,
                    random_seed=random_seed,
                    stage=stage,
                    dataset_builder=dataset_partition,
                )
                worker_steps.append(ret)

            else:
                ret = worker.build_dataset_from_csv(
                    file_path=train_dict[device],
                    label=label if self.device_y == device else None,
                    s_w=s_w_partitions,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    repeat_count=epochs,
                    random_seed=random_seed,
                    stage=stage,
                )
                worker_steps.append(ret)
        worker_steps_per_epoch = reveal(worker_steps)
        worker_steps_per_epoch = [
            steps for steps in worker_steps_per_epoch if steps != -1
        ]
        assert len(set(worker_steps_per_epoch)) == 1, (
            "steps_per_epoch of all parties must be same, "
            "Please check whether the batchsize or steps_per_epoch of all parties are consistent"
        )
        steps_per_epoch = worker_steps_per_epoch[0]
        self._workers[self.device_y].set_steps_per_epoch(steps_per_epoch)

        return steps_per_epoch

    def fit(
        self,
        x: Union[
            VDataFrame,
            FedNdarray,
            List[Union[HDataFrame, VDataFrame, FedNdarray]],
            Dict[PYU, str],
        ],
        y: Union[VDataFrame, FedNdarray, PYUObject, str],
        batch_size=32,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_data: Tuple = None,
        shuffle=False,
        sample_weight=None,
        validation_freq=1,
        dp_spent_step_freq=None,
        dataset_builder: Callable[[List], Tuple[int, Iterable]] = None,
        audit_log_dir: str = None,
        audit_log_params: dict = {},
        early_stopping_batch_step: int = 0,
        early_stopping_warmup_step: int = 0,
        random_seed: int = None,
    ):
        """Vertical split learning training interface

        Args:
            x: Input data. It could be:

            - VDataFrame: a vertically aligned dataframe.
            - FedNdArray: a vertically aligned ndarray.
            - List[Union[HDataFrame, VDataFrame, FedNdarray]]: list of dataframe or ndarray.
            - Dict[PYU, str]: dict of PYU and its input data file path, for performance, dtypes need to be provided.

            y: Target data. It could be:
            - VDataFrame.
            - FedNdarray which has only one partition.
            - PYUObject.
            - str: when x is a Dict file path, y is the label name of the input dataset.

            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train the model
            verbose: 0, 1. Verbosity mode
            callbacks: List of Callback or Dict[device, Callback]. Callback can be secretflow.ml.nn.callbacks:
            validation_data: of Data on which to validate, it could be:
            - A tuple of (test_data, test_label) with the data of type VDataFrame or FedNdarray.
            - A tuple of (test_data, test_label, valid_sample_weight)
            - A tuple of (Dict[PYU, str], str) when requires the file inputs just like x and y..
            shuffle: Whether shuffle dataset or not
            validation_freq: specifies how many training epochs to run before a new validation run is performed
            sample_weight: weights for the training samples
            dp_spent_step_freq: specifies how many training steps to check the budget of dp
            dataset_builder: Callable function, its input is `x` or `[x, y]` if y is set, it should return a
                dataset.
            audit_log_dir: If audit_log_dir is set, audit model will be enabled
            audit_log_params: Kwargs for saving audit model, eg: {'save_traces'=True, 'save_format'='h5'}
            random_seed: seed for prg, will only affect dataset shuffle
            dtypes: Dict[PYU, Dict[str, Dtype]]
        """
        if random_seed is None:
            random_seed = global_random(self.device_y, 100000)

        # record parameters
        params = locals()
        logging.info(f"SL Train Params: {params}")

        # sanity check
        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), f"batch_size should be integer > 0"
        assert isinstance(validation_freq, int) and validation_freq >= 1
        if dp_spent_step_freq is not None:
            assert isinstance(dp_spent_step_freq, int) and dp_spent_step_freq >= 1

        # get basenet ouput num
        self.basenet_output_num = {
            device: reveal(worker.get_basenet_output_num())
            for device, worker in self._workers.items()
        }
        self.agglayer.set_basenet_output_num(self.basenet_output_num)

        # build dataset
        train_x, train_y = x, y
        if isinstance(train_x, Dict):
            assert isinstance(train_y, str), (
                f"When the input x is type of Dict, the data will read from files, "
                f"and y must be a label name with type str."
            )
            steps_per_epoch = self.handle_file(
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
        else:
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
        if validation_data is not None:
            assert isinstance(validation_data, tuple) and (
                len(validation_data) == 2 or len(validation_data) == 3
            ), f"validation_data must be tuple type with (test_x, test_y) or (test_x, test_y, valid_sample_weight)."
            validation = True
            logging.debug("validation_data provided")
            if len(validation_data) == 2:
                valid_x, valid_y = validation_data
                valid_sample_weight = None
            else:
                valid_x, valid_y, valid_sample_weight = validation_data
            assert (
                valid_x is not None and valid_y is not None
            ), f"Neither x nor y in validation data cannot be None."
            if isinstance(valid_x, Dict):
                assert isinstance(valid_y, str), (
                    f"When the input x is type of Dict, the data will read from files, "
                    f"and y must be a label name with type str."
                )
                valid_steps = self.handle_file(
                    valid_x,
                    valid_y,
                    sample_weight=valid_sample_weight,
                    batch_size=batch_size,
                    epochs=epochs,
                    stage="eval",
                    dataset_builder=dataset_builder,
                )
            else:
                valid_steps = self.handle_data(
                    valid_x,
                    valid_y,
                    sample_weight=valid_sample_weight,
                    batch_size=batch_size,
                    epochs=epochs,
                    stage="eval",
                    dataset_builder=dataset_builder,
                )
        else:
            valid_x, valid_y, valid_sample_weight = None, None, None

        wait_steps = min(min(self.get_cpus()) * 2, 100)
        # setup callback list
        callbacks = CallbackList(
            callbacks=callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            workers=self._workers,
            device_y=self.device_y,
            epochs=epochs,
            verbose=verbose,
            steps=steps_per_epoch,
        )

        callbacks.on_train_begin()
        for epoch in range(epochs):
            res = []
            report_list = []
            report_list.append(f"epoch: {epoch+1}/{epochs} - ")
            self._workers[self.device_y].reset_metrics()
            callbacks.on_epoch_begin(epoch=epoch)

            hiddens_buf = [None] * (self.pipeline_size - 1)
            for step in range(0, steps_per_epoch + self.pipeline_size - 1):
                if step < steps_per_epoch:
                    hiddens = {}
                    callbacks.on_train_batch_begin(step)
                    for device, worker in self._workers.items():
                        # 1. Local calculation of basenet
                        hidden = worker.base_forward(stage="train", step=step)
                        hiddens[device] = hidden
                    hiddens_buf.append(hiddens)
                # clean up buffer
                hiddens = hiddens_buf.pop(0)
                # Async transfer hiddens to label side
                if hiddens is None:
                    continue
                # During pipeline strategy, the backpropagation process of the model will lag n cycles behind the forward propagation process.
                step = step - self.pipeline_size + 1

                # do agglayer forward
                agg_hiddens = self.agglayer.forward(hiddens)

                # 3. Fusenet do local calculates and return gradients
                gradients = self._workers[self.device_y].fuse_net(agg_hiddens)

                # In some strategies, we need to bypass the backpropagation step.
                skip_gradient = False
                if self.check_skip_grad:
                    skip_gradient = reveal(
                        self._workers[self.device_y].get_skip_gradient()
                    )

                if not skip_gradient:
                    # do agglayer backward
                    scatter_gradients = self.agglayer.backward(gradients)
                    for device, worker in self._workers.items():
                        if device in scatter_gradients.keys():
                            worker.base_backward(scatter_gradients[device])

                # for EarlyStoppingBatch, evalute model every early_stopping_batch_step
                if (
                    early_stopping_batch_step > 0
                    and step > early_stopping_warmup_step
                    and step % early_stopping_batch_step == 0
                ):
                    wait(res)
                    res = []
                    # as evaluation will change metrics' state(training stage),
                    # we temporarily save it here, and recover later
                    self._workers[self.device_y].staging_metric_states()

                    # validation
                    self._workers[self.device_y].reset_metrics()

                    callbacks.on_test_begin()
                    res = []
                    for val_step in range(0, valid_steps):
                        callbacks.on_test_batch_begin(batch=val_step)
                        hiddens = {}  # driver end
                        for device, worker in self._workers.items():
                            hidden = worker.base_forward("eval", step=val_step)
                            hiddens[device] = hidden
                        agg_hiddens = self.agglayer.forward(hiddens, axis=0)

                        metrics = self._workers[self.device_y].evaluate(agg_hiddens)
                        res.append(metrics)
                        if len(res) == wait_steps:
                            wait(res)
                            res = []
                        callbacks.on_test_batch_end(batch=val_step)
                    wait(res)
                    callbacks.on_test_end(metrics)

                    callbacks.on_train_batch_end(step)

                    # recover metrics's state(training stage)
                    self._workers[self.device_y].recover_metric_states()

                    if callbacks.stop_training[0]:
                        break
                else:
                    callbacks.on_train_batch_end(step)

                res.append(gradients)

                if self.dp_strategy_dict is not None and dp_spent_step_freq is not None:
                    current_step = epoch * steps_per_epoch + step
                    if current_step % dp_spent_step_freq == 0:
                        privacy_device = {}
                        for device, dp_strategy in self.dp_strategy_dict.items():
                            privacy_dict = dp_strategy.get_privacy_spent(current_step)
                            privacy_device[device] = privacy_dict
                if len(res) == wait_steps:
                    wait(res)
                    res = []
            assert (
                len(hiddens_buf) == 0
            ), f'hiddens buffer unfinished, len: {len(hiddens_buf)}'
            if validation and epoch % validation_freq == 0:
                callbacks.on_test_begin()
                # validation
                self._workers[self.device_y].reset_metrics()

                res = []
                for step in range(0, valid_steps):
                    callbacks.on_test_batch_begin(batch=step)
                    hiddens = {}  # driver end
                    for device, worker in self._workers.items():
                        hidden = worker.base_forward("eval", step=step)
                        hiddens[device] = hidden
                    agg_hiddens = self.agglayer.forward(hiddens)

                    metrics = self._workers[self.device_y].evaluate(agg_hiddens)
                    res.append(metrics)
                    if len(res) == wait_steps:
                        wait(res)
                        res = []
                    callbacks.on_test_batch_end(batch=step)
                wait(res)
                callbacks.on_test_end(metrics)
                # save checkpoint
                # TODO: Use checkpoint callback rewrite this logic @Juxing
                if audit_log_dir is not None:
                    epoch_base_model_path = {
                        device: os.path.join(
                            audit_log_dir,
                            "base_model",
                            device.party,
                            str(epoch),
                        )
                        for device in self._workers.keys()
                    }
                    epoch_fuse_model_path = os.path.join(
                        audit_log_dir,
                        "fuse_model",
                        str(epoch),
                    )
                    self.save_model(
                        base_model_path=epoch_base_model_path,
                        fuse_model_path=epoch_fuse_model_path,
                        is_test=self.simulation,
                        **audit_log_params,
                    )
            callbacks.on_epoch_end(epoch)
            # TODO: Need EarlystopCallback
            if callbacks.stop_training[0]:
                break

        callbacks.on_train_end()
        return callbacks.history

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
        callbacks=None,
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
            callbacks: List of `keras.callbacks.Callback` instances.
        """

        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), f"batch_size should be integer > 0"

        predict_steps = self.handle_data(
            x,
            None,
            batch_size=batch_size,
            stage="eval",
            epochs=1,
            dataset_builder=dataset_builder,
        )

        # setup callback list
        callbacks = CallbackList(
            callbacks=callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            workers=self._workers,
            device_y=self.device_y,
            verbose=verbose,
            steps=predict_steps,
        )
        result = []
        wait_steps = min(min(self.get_cpus()) * 2, 100)
        res = []
        callbacks.on_predict_begin()
        for step in range(0, predict_steps):
            callbacks.on_predict_batch_begin(step)
            forward_data_dict = {}
            for device, worker in self._workers.items():
                if device not in self.base_model_dict:
                    continue
                f_data = worker.base_forward(stage="eval", step=step)
                forward_data_dict[device] = f_data

            agg_hiddens = self.agglayer.forward(forward_data_dict)

            y_pred = self._workers[self.device_y].predict(agg_hiddens)
            result.append(y_pred)

            callbacks.on_predict_batch_end(batch=step)
            res.append(y_pred)
            if len(res) == wait_steps:
                wait(res)
                res = []
        wait(res)
        callbacks.on_predict_end()
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
        dataset_builder: Dict = None,
        random_seed: int = None,
        callbacks=None,
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
            dataset_builder: Callable function, its input is `x` or `[x, y]` if y is set, it should return dataset.
            random_seed: Seed for prgs, will only affect shuffle
            callbacks: List of `secretflow.ml.nn.callbacks.Callback` instances.
        Returns:
            metrics: federate evaluate result
        """

        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), f"batch_size should be integer > 0"

        if random_seed is None:
            random_seed = global_random(self.device_y, 100000)
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

        callbacks = CallbackList(
            callbacks=callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            workers=self._workers,
            device_y=self.device_y,
            verbose=verbose,
            steps=evaluate_steps,
        )
        callbacks.on_test_begin()
        wait_steps = min(min(self.get_cpus()) * 2, 100)
        for step in range(0, evaluate_steps):
            callbacks.on_test_batch_begin(step)
            hiddens = {}  # driver端
            for device, worker in self._workers.items():
                hidden = worker.base_forward(stage="eval", step=step)
                hiddens[device] = hidden

            agg_hiddens = self.agglayer.forward(hiddens)

            metrics = self._workers[self.device_y].evaluate(agg_hiddens)
            if (step + 1) % wait_steps == 0:
                wait(metrics)
            callbacks.on_test_batch_end(batch=step)

        callbacks.on_test_end(metrics)
        return metrics

    def save_model(
        self,
        base_model_path: Union[str, Dict[PYU, str]] = None,
        fuse_model_path: str = None,
        is_test=False,
        **kwargs,
    ):
        """Vertical split learning save model interface

        Args:
            base_model_path: base model path,only support format like 'a/b/c', where c is the model name
            fuse_model_path: fuse model path
            is_test: whether is test mode
            kwargs: other argument inherit from tf or torch
        Examples:
            >>> save_params = {'save_traces' : True,
            >>>                'save_format' : 'h5',}
            >>> slmodel.save_model(base_model_path,
            >>>                    fuse_model_path,)
            >>>                    is_test=True,)
            >>> # just passing params in
            >>> slmodel.save_model(base_model_path,
            >>>                    fuse_model_path,
            >>>                    is_test=True,
            >>>                    save_traces=True,
            >>>                    save_format='h5')
        """
        assert isinstance(
            base_model_path, (str, Dict)
        ), f'Model path accepts string or dict but got {type(base_model_path)}.'
        assert fuse_model_path is not None, "Fuse model path cannot be empty"
        if isinstance(base_model_path, str):
            base_model_path = {
                device: base_model_path for device in self.base_model_dict.keys()
            }

        res = []
        for device, worker in self._workers.items():
            if device not in self.base_model_dict.keys():
                continue
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
                    os.path.join(base_model_dir, base_model_name), **kwargs
                )
            )

        res.append(
            self._workers[self.device_y].save_fuse_model(fuse_model_path, **kwargs)
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
                device: base_model_path for device in self.base_model_dict.keys()
            }
        res = []
        for device, worker in self._workers.items():
            if device not in self.base_model_dict.keys():
                continue
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

    def export_model(
        self,
        base_model_path: Union[str, Dict[PYU, str]] = None,
        fuse_model_path: str = None,
        save_format="tf",
        is_test=False,
        **kwargs,
    ):
        """Vertical split learning export model interface

        Args:
            base_model_path: base model path,only support format like 'a/b/c', where c is the model name
            fuse_model_path: fuse model path
            save_format: what format to export
            kwargs: other argument inherit from onnx safer
        """
        assert isinstance(
            base_model_path, (str, Dict)
        ), f'Model path accepts string or dict but got {type(base_model_path)}.'
        assert fuse_model_path is not None, "Fuse model path cannot be empty"
        if isinstance(base_model_path, str):
            base_model_path = {
                device: base_model_path for device in self.base_model_dict.keys()
            }

        res = []
        base_input_output_infos = {}
        for device, worker in self._workers.items():
            if device not in self.base_model_dict.keys():
                continue
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

            input_output_info = worker.export_base_model(
                os.path.join(base_model_dir, base_model_name),
                save_format=save_format,
                **kwargs,
            )
            res.append(input_output_info)
            base_input_output_infos[device] = input_output_info

        fuse_input_output_info = self._workers[self.device_y].export_fuse_model(
            fuse_model_path, save_format=save_format, **kwargs
        )
        res.append(fuse_input_output_info)
        wait(res)
        return base_input_output_infos, fuse_input_output_info

    def get_cpus(self) -> List[int]:
        return reveal([device(lambda: cpu_count())() for device in self._workers])
