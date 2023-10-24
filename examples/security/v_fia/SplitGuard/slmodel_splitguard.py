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

# The code is finished by Haodong Zhao. Mail: zhaohaodong@sjtu.edu.cn
# The code is modified from sl_model.py, and Split Guard detection from the paper <SplitGuard: Detecting and Mitigating Training-Hijacking Attacks in Split Learning>  is added. The code of https://github.com/ege-erdogan/splitguard is referenced.


"""SLModel

"""
import logging
import math
import os
import random
from typing import Callable, Dict, Iterable, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

import secretflow as sf
from secretflow.data import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, Device, reveal, wait
from secretflow.device.device.pyu import PYUObject
from secretflow.ml.nn import SLModel
from secretflow.ml.nn.sl.agglayer.agg_layer import AggLayer
from secretflow.ml.nn.sl.agglayer.agg_method import AggMethod
from secretflow.ml.nn.sl.strategy_dispatcher import dispatch_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.utils.random import global_random


# Add functions to compute SG Score
def angle(v1, v2):
    """Function to compute between two vectors, denoted as theta in paper."""
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


# Add functions to compute SG Score
def sigmoid(S, alpha=1, beta=1):
    """Squahing function sigmoid in paper."""
    S_p = S * alpha
    return (1 / (1 + np.exp(-S_p))) ** beta


def sg_score(fakes, R1, R2, alpha=1, beta=1):
    """Function to compute SG Score.

    return SG Score
    """
    F_mean = sum(fakes) / len(fakes)  # mean of F
    R1_mean = sum(R1) / len(R1)  # mean of R_1
    R2_mean = sum(R2) / len(R2)  # mean of R_2
    R_mean = (sum(R1) + sum(R2)) / (len(R1) + len(R2))  # mean of R

    F_mean_mag = sum([np.linalg.norm(v) for v in fakes]) / len(fakes)  # mean of ||F||
    R1_mean_mag = sum([np.linalg.norm(v) for v in R1]) / len(R1)  # mean of ||R_1||
    R2_mean_mag = sum([np.linalg.norm(v) for v in R2]) / len(R2)  # mean of ||R_2||
    R_mean_mag = (
        sum([np.linalg.norm(v) for v in R1]) + sum([np.linalg.norm(v) for v in R2])
    ) / (
        len(R1) + len(R2)
    )  # mean of ||R||    R_mean_mag=(R1_mean_mag+R2_mean_mag)/2

    # d(F,R)+d(R_1,R_2)
    mag_div = abs(F_mean_mag - R_mean_mag) + abs(R1_mean_mag - R2_mean_mag)

    # compute S as in equation (6). The angle of (F_mean,R_mean) is the same as the angle of (F_sum,R_sum).
    S = angle(F_mean, R_mean) * (abs(F_mean_mag - R_mean_mag) / mag_div) - angle(
        R1_mean, R2_mean
    ) * (abs(R2_mean_mag - R1_mean_mag) / mag_div)

    return sigmoid(S, alpha=alpha, beta=beta)


# Modified from the class SLModel. Split Guard is added by modify the fit() function, and plot_sgscore() function is added to visualize results.
class SLModel_SG(SLModel):
    def __init__(
        self,
        base_model_dict: Dict[Device, Callable[[], "tensorflow.keras.Model"]] = {},
        device_y: PYU = None,
        model_fuse: Callable[[], "tensorflow.keras.Model"] = None,
        dp_strategy_dict: Dict[Device, DPStrategy] = None,
        random_seed: int = None,
        backend: str = "tensorflow",
        strategy="split_nn",
        agg_method: AggMethod = None,
        N: int = 20,
        alpha: float = 1.0,
        beta: int = 1,
        fake_batch: List = [],
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
            device_agg: The party do aggregation,it can be a PYU,SPU,etc.
        """
        super().__init__(
            base_model_dict,
            device_y,
            model_fuse,
            dp_strategy_dict,
            random_seed,
            backend,
            strategy,
            agg_method,
            **kwargs,
        )
        self.alhpa = alpha
        self.beta = beta
        self.N = N
        self.fake_batch = fake_batch
        # Add data to compute SG Score
        self.fake, self.r_1, self.r_2, self.results, self.fake_indices = (
            [],
            [],
            [],
            [],
            [],
        )

    # Add function to convert data format to ndarray.
    @staticmethod
    def convert_to_ndarray(*data: List) -> Union[List[jnp.ndarray], jnp.ndarray]:
        def _convert_to_ndarray(hidden):
            # processing data
            if not isinstance(hidden, jnp.ndarray):
                if isinstance(hidden, (tf.Tensor, torch.Tensor)):
                    hidden = jnp.array(hidden.numpy())
                if isinstance(hidden, np.ndarray):
                    hidden = jnp.array(hidden)
            return hidden

        if isinstance(data, Tuple) and len(data) == 1:
            # The case is after packing and unpacking using PYU, a tuple of length 1 will be obtained, if 'num_return' is not specified to PYU.
            data = data[0]
        if isinstance(data, (List, Tuple)):
            return [_convert_to_ndarray(d) for d in data]
        else:
            return _convert_to_ndarray(data)

    # Add function to convert data format to tensor.
    @staticmethod
    def convert_to_tensor(hidden: Union[List, Tuple], backend: str):
        if backend == "tensorflow":
            if isinstance(hidden, (List, Tuple)):
                hidden = [tf.convert_to_tensor(d) for d in hidden]

            else:
                hidden = tf.convert_to_tensor(hidden)
        elif backend == "torch":
            if isinstance(hidden, (List, Tuple)):
                hidden = [torch.Tensor(d) for d in hidden]
            else:
                hidden = torch.Tensor(hidden)
        return hidden

    def fit(
        self,
        x: Union[
            VDataFrame, FedNdarray, List[Union[HDataFrame, VDataFrame, FedNdarray]],
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
        audit_log_params: dict = {},
        random_seed: int = None,
        callback: Callable = None,
    ):
        """Vertical split learning training interface

        Split Guard code is added.

        Args:
            x: Input data. It could be:

            - VDataFrame: a vertically aligned dataframe.
            - FedNdArray: a vertically aligned ndarray.
            - List[Union[HDataFrame, VDataFrame, FedNdarray]]: list of dataframe or ndarray.

            y: Target data. It could be a VDataFrame or FedNdarray which has only one partition, or a PYUObject.
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train the model
            verbose: 0, 1. Verbosity mode
            callbacks: List of Callback or Dict[device, Callback]. Callback can be:
            - `keras.callbacks.Callback` for tensorflow backend
            - `secretflow.ml.nn.sl.backend.torch.callback.Callback` for torch backend
            validation_data: Data on which to validate
            shuffle: Whether shuffle dataset or not
            validation_freq: specifies how many training epochs to run before a new validation run is performed
            sample_weight: weights for the training samples
            dp_spent_step_freq: specifies how many training steps to check the budget of dp
            dataset_builder: Callable function, its input is `x` or `[x, y]` if y is set, it should return a
                dataset.
            audit_log_dir: If audit_log_dir is set, audit model will be enabled
            audit_log_params: Kwargs for saving audit model, eg: {'save_traces'=True, 'save_format'='h5'}
            random_seed: seed for prg, will only affect dataset shuffle
        """
        if random_seed is None:
            random_seed = global_random(self.device_y, 100000)

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
        self.agglayer.set_basenet_output_num(self.basenet_output_num)
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
        if isinstance(callbacks, dict):
            for dev, callback_builder in callbacks.items():
                self._workers[dev].init_training(callback_builder, epochs=epochs)
        else:
            self._workers[self.device_y].init_training(callbacks, epochs=epochs)
        [worker.on_train_begin() for worker in self._workers.values()]
        wait_steps = min(min(self.get_cpus()) * 2, 100)
        for epoch in range(epochs):
            res = []
            report_list = []
            report_list.append(f"epoch: {epoch+1}/{epochs} - ")
            if verbose == 1:
                pbar = tqdm(total=steps_per_epoch)
            self._workers[self.device_y].reset_metrics()
            [worker.on_epoch_begin(epoch) for worker in self._workers.values()]

            hiddens_buf = [None] * (self.pipeline_size - 1)
            for step in range(0, steps_per_epoch + self.pipeline_size - 1):
                if step < steps_per_epoch:
                    if verbose == 1:
                        pbar.update(1)
                    hiddens = {}
                    self._workers[self.device_y].on_train_batch_begin(step=step)
                    for device, worker in self._workers.items():
                        # 1. Local calculation of basenet
                        hidden = worker.base_forward(stage="train")
                        # 2. The results of basenet are sent to fusenet

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
                agg_hiddens = self.agglayer.forward(hiddens, axis=0)

                # 3. Fusenet do local calculates and return gradients
                gradients = self._workers[self.device_y].fuse_net(agg_hiddens)
                ######################
                # Add code to perform Split Guard here
                skip_gradient = self.callback(step, steps_per_epoch, gradients)

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

                r_count = self._workers[self.device_y].on_train_batch_end(step=step)
                res.append(r_count)
                [
                    worker.on_train_batch_end(step=step)
                    for dev, worker in self._workers.items()
                    if dev != self.device_y
                ]

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
            ), f"hiddens buffer unfinished, len: {len(hiddens_buf)}"
            if validation and epoch % validation_freq == 0:
                # validation
                self._workers[self.device_y].reset_metrics()
                res = []
                for step in range(0, valid_steps):
                    hiddens = {}  # driver end
                    for device, worker in self._workers.items():
                        hidden = worker.base_forward("eval")
                        hiddens[device] = hidden
                    agg_hiddens = self.agglayer.forward(hiddens, axis=0)

                    metrics = self._workers[self.device_y].evaluate(agg_hiddens)
                    res.append(metrics)
                    if len(res) == wait_steps:
                        wait(res)
                        res = []
                wait(res)
                self._workers[self.device_y].on_validation(metrics)

                # save checkpoint
                if audit_log_dir is not None:
                    epoch_base_model_path = {
                        device: os.path.join(
                            audit_log_dir, "base_model", device.party, str(epoch),
                        )
                        for device in self._workers.keys()
                    }
                    epoch_fuse_model_path = os.path.join(
                        audit_log_dir, "fuse_model", str(epoch),
                    )
                    self.save_model(
                        base_model_path=epoch_base_model_path,
                        fuse_model_path=epoch_fuse_model_path,
                        is_test=self.simulation,
                        **audit_log_params,
                    )
            epoch_log = self._workers[self.device_y].on_epoch_end(epoch)
            call_res = [
                worker.on_epoch_end(epoch)
                for dev, worker in self._workers.items()
                if dev != self.device_y
            ]
            wait(call_res)
            for name, metric in reveal(epoch_log).items():
                report_list.append(f"{name}:{metric} ")
            report = " ".join(report_list)
            if verbose == 1:
                pbar.set_postfix_str(report)
                pbar.close()
            if reveal(self._workers[self.device_y].get_stop_training()):
                break

        history = self._workers[self.device_y].on_train_end()
        call_res = [
            worker.on_train_end()
            for dev, worker in self._workers.items()
            if dev != self.device_y
        ]
        wait(call_res)
        return reveal(history)

    def plot_sgscore(self):
        """Function to plot fig according to SG Score computed."""
        import matplotlib.pyplot as plt

        plt.plot(self.results, label="detection")
        plt.ylim(0, 1.1)
        plt.xlabel("No. of fake batches")
        plt.ylabel("SG score")
        plt.legend()
        plt.savefig("sf-sgscore.png")
        plt.show()

    def callback(self, step, steps_per_epoch, gradients):
        if step < (steps_per_epoch + self.pipeline_size - 2):
            scatter_gradients = self.agglayer.backward(gradients)
            worker_list = list(self.base_model_dict.keys())
            client_device = worker_list[0]
            # Get client gradient
            grad_client = scatter_gradients[client_device]
            grad_client_np = client_device(self.convert_to_ndarray)(grad_client)
            client_grad = sf.reveal(grad_client_np)[0][0].flatten()

            # Fake batches
            if step in self.fake_batch:
                self.fake.append(client_grad)
                if len(self.r_1) > 0 and len(self.r_2) > 0 and len(self.fake) > 0:
                    sg = sg_score(
                        self.fake, self.r_1, self.r_2, alpha=self.alhpa, beta=self.beta,
                    )
                    self.results.append(sg)
            # Regular batches
            else:
                if step > self.N:
                    if random.random() <= 0.5:
                        self.r_1.append(client_grad)
                    else:
                        self.r_2.append(client_grad)

        # In some strategies, we need to bypass the backpropagation step.
        skip_gradient = False
        if step in self.fake_batch:
            skip_gradient = True
        return skip_gradient
        # End of Split Guard code
