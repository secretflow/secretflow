import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from manager import BaseManager
from sklearn.metrics import auc, roc_auc_score, roc_curve

"""SLModel

"""
import logging
import math
import os
from typing import Callable, Dict, Iterable, List, Tuple, Union

from multiprocess import cpu_count
from tqdm import tqdm

import secretflow as sf
from secretflow.data.base import Partition
from secretflow.data.horizontal import HDataFrame
from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import PYU, Device, reveal, wait
from secretflow.device.device.pyu import PYUObject
from secretflow.ml.nn.sl.agglayer.agg_layer import AggLayer
from secretflow.ml.nn.sl.agglayer.agg_method import AggMethod
from secretflow.ml.nn.sl.strategy_dispatcher import dispatch_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.utils.random import global_random


# torch实现的原版Norm Attack
def attach_normattack_to_splitnn(
    cls, attack_criterion, target_client_index=0, device="cpu"
):
    class NormAttackSplitNNWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(NormAttackSplitNNWrapper, self).__init__(*args, **kwargs)
            self.attack_criterion = attack_criterion
            self.target_client_index = target_client_index
            self.device = device

        def extract_intermidiate_gradient(self, outputs):
            self.backward_gradient(outputs.grad)
            return self.clients[self.target_client_index].grad_from_next_client

        def attack(self, dataloader):
            """Culculate leak_auc on the given SplitNN model
            reference: https://arxiv.org/abs/2102.08504
            Args:
                dataloader (torch dataloader): dataloader for evaluation
                criterion: loss function for training
                device: cpu or GPU
            Returns:
                score: culculated leak auc
            """
            epoch_labels = []
            epoch_g_norm = []
            epoch_g_norm_2 = []
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = self.attack_criterion(outputs, labels)
                loss.backward()

                grad_from_server = self.extract_intermidiate_gradient(outputs)
                g_norm = grad_from_server.pow(2).sum(dim=1).sqrt()
                g_norm_2 = grad_from_server.abs().sum(dim=1)

                epoch_labels.append(labels)
                epoch_g_norm.append(g_norm)
                epoch_g_norm_2.append(g_norm_2)

            epoch_labels = torch.cat(epoch_labels)
            epoch_g_norm = torch.cat(epoch_g_norm)
            epoch_g_norm_2 = torch.cat(epoch_g_norm_2)

            score = roc_auc_score(epoch_labels, epoch_g_norm.view(-1, 1))
            score_2 = roc_auc_score(epoch_labels, epoch_g_norm_2.view(-1, 1))

            fpr, tpr, thersholds = roc_curve(epoch_labels, epoch_g_norm.view(-1, 1))
            fpr_2, tpr_2, thersholds_2 = roc_curve(
                epoch_labels, epoch_g_norm_2.view(-1, 1)
            )
            # print(epoch_labels)
            # print(epoch_g_norm)

            roc_auc = auc(fpr, tpr)
            roc_auc_2 = auc(fpr_2, tpr_2)

            plt.plot(
                fpr, tpr, "k-.", label="ROC (area = {0:.2f})".format(roc_auc), lw=2
            )
            plt.plot(
                fpr_2,
                tpr_2,
                "b--",
                label="ROC (area = {0:.2f})".format(roc_auc_2),
                lw=2,
            )

            plt.xlim(
                [-0.05, 1.05]
            )  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
            plt.ylim([-0.05, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")  # 可以使用中文，但需要导入一些库即字体
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.show()

            return score

    return NormAttackSplitNNWrapper


class NormAttackSplitNNManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_normattack_to_splitnn(cls, *self.args, **self.kwargs)


# SecretFlow版的Norm Attack
def attach_normattack_to_splitnn_sf(
    cls, attack_criterion=None, target_client_index=0, device="cpu"
):
    class NormAttackSplitNNWrapper_sf(cls):
        def __init__(self, *args, **kwargs):
            super(NormAttackSplitNNWrapper_sf, self).__init__(*args, **kwargs)
            self.attack_criterion = attack_criterion
            self.target_client_index = target_client_index
            self.device = device

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

        def extract_intermidiate_gradient(self, outputs):
            self.backward_gradient(outputs.grad)
            return self.clients[self.target_client_index].grad_from_next_client

        def attack_grad(
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
            audit_log_params: dict = {},
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

            self._workers[self.device_y].init_training(callbacks, epochs=epochs)
            [worker.on_train_begin() for worker in self._workers.values()]
            wait_steps = min(min(self.get_cpus()) * 2, 100)

            epoch_labels = []
            epoch_g_norm = []
            for epoch in range(epochs):
                res = []
                report_list = []
                report_list.append(f"epoch: {epoch+1}/{epochs} - ")
                if verbose == 1:
                    pbar = tqdm(total=steps_per_epoch)
                self._workers[self.device_y].reset_metrics()
                [worker.on_epoch_begin(epoch) for worker in self._workers.values()]

                for step in range(0, steps_per_epoch):
                    if verbose == 1:
                        pbar.update(1)
                    hiddens = {}
                    self._workers[self.device_y].on_train_batch_begin(step=step)
                    for device, worker in self._workers.items():
                        # 1. Local calculation of basenet
                        hidden = worker.base_forward(stage="train")
                        # 2. The results of basenet are sent to fusenet
                        hiddens[device] = hidden
                    # do agglayer forward
                    agg_hiddens = self.agglayer.forward(hiddens, axis=0)
                    if isinstance(agg_hiddens, PYUObject):
                        agg_hiddens = [agg_hiddens]
                    # 3. Fusenet do local calculates and return gradients
                    gradients = self._workers[self.device_y].fuse_net(*agg_hiddens)
                    scatter_gradients = self.agglayer.backward(gradients)

                    worker_list = list(self.base_model_dict.keys())
                    client_device = worker_list[0]
                    grad_client = scatter_gradients[client_device]
                    grad_client_np = client_device(self.convert_to_ndarray)(grad_client)
                    grad_np = sf.reveal(grad_client_np)[0]

                    grad_norm_np = jnp.sqrt(jnp.sum(jnp.square(grad_np), axis=1))

                    epoch_g_norm.append(grad_norm_np)

                return epoch_g_norm

    return NormAttackSplitNNWrapper_sf


class NormAttackSplitNNManager_sf(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_normattack_to_splitnn_sf(cls, *self.args, **self.kwargs)
