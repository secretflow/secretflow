#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# # Copyright 2022 Ant Group Co., Ltd.
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


"""sl model base
"""
import copy
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tensorflow.python.keras import callbacks as callbacks_module
from torch import nn
from torch.nn.modules.loss import _Loss as BaseTorchLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import secretflow.device as ft
from secretflow.device import PYUObject, proxy
from secretflow.security.privacy import DPStrategy


class SLBaseModel(ABC):
    def __init__(self, builder_base: Callable, builder_fuse: Callable = None):
        self.model_base = builder_base() if builder_base is not None else None
        self.model_fuse = builder_fuse() if builder_fuse is not None else None

    @abstractmethod
    def build_dataset(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        s_w: Optional[np.ndarray] = None,
        batch_size=32,
        buffer_size=128,
        repeat_count=1,
    ):
        pass

    @abstractmethod
    def base_forward(self):
        pass

    @abstractmethod
    def base_backward(self):
        pass

    @abstractmethod
    def fuse_net(self, hiddens):
        pass


class SLBaseModule(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self, parameters=None):
        if parameters is None:
            parameters = self.parameters()
        grads = []
        for p in parameters:
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(
        self,
        gradients: List[Union[torch.Tensor, np.ndarray]],
        parameters: Optional[List[torch.Tensor]] = None,
    ):
        if parameters is None:
            parameters = self.parameters()
        for g, p in zip(gradients, parameters):
            if g is not None:
                p.grad = torch.from_numpy(g.copy())


class SLBaseTFModel(SLBaseModel):
    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        builder_fuse: Callable[[], tf.keras.Model],
        dp_strategy: DPStrategy,
    ):
        super().__init__(builder_base, builder_fuse)

        self.dp_strategy = dp_strategy
        self.embedding_dp = (
            self.dp_strategy.embedding_dp if dp_strategy is not None else None
        )
        self.label_dp = self.dp_strategy.label_dp if dp_strategy is not None else None

        self.train_set = None
        self.eval_set = None
        self.valid_set = None
        self.tape = None
        self.h = None
        self.train_x, self.train_y = None, None
        self.eval_x, self.eval_y = None, None
        self.kwargs = {}
        self.has_y = False
        self.has_s_w = False
        self.train_sample_weight = None
        self.eval_sample_weight = None
        self.fuse_callbacks = None
        self.logs = None
        self.epoch_logs = None
        self.training_logs = None
        self.steps_per_epoch = None

    def build_base_model(self, builder: Callable, *args, **kwargs):
        self.model_base = builder(*args, **kwargs)

    def build_fuse_model(self, builder: Callable, *args, **kwargs):
        self.model_fuse = builder(*args, **kwargs)

    @staticmethod
    @tf.custom_gradient
    def fuse_op(x, y):
        def grad(upstream):
            return upstream * y, upstream * y

        return x, grad

    def init_data(self):
        self.train_x, self.train_y = None, None
        self.eval_x, self.eval_y = None, None
        self.train_sample_weight = None
        self.eval_sample_weight = None

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch

    def build_dataset(
        self,
        *x: List[np.ndarray],
        y: Optional[np.ndarray] = None,
        s_w: Optional[np.ndarray] = None,
        batch_size=32,
        buffer_size=128,
        shuffle=False,
        repeat_count=1,
        stage="train",
        random_seed=1234,
        dataset_builder: Callable = None,
    ):
        """build tf.data.Dataset

        Args:
            x: feature, FedNdArray or HDataFrame
            y: label, FedNdArray or HDataFrame
            s_w: sample weight, FedNdArray or HDataFrame
            batch_size: Number of samples per gradient update
            buffer_size: buffer size for shuffling
            shuffle: whether shuffle the dataset or not
            repeat_count: num of repeats
            stage: stage of this datset
            random_seed: Prg seed for shuffling
        """
        assert len(x) > 0 or x[0] is not None, "X can not be None, please check"
        x = [xi for xi in x]
        self.has_y = False
        self.has_s_w = False
        if y is not None and len(y.shape) > 0:
            self.has_y = True
            # Label differential privacy
            x.append(
                self.label_dp(y)
                if stage == "train" and self.label_dp is not None
                else y
            )
            if s_w is not None and len(s_w.shape) > 0:
                self.has_s_w = True
                x.append(s_w)

        steps_per_epoch = None
        if dataset_builder is None:
            # convert pandas.DataFrame to numpy.ndarray
            x = [t.values if isinstance(t, pd.DataFrame) else t for t in x]
            # https://github.com/tensorflow/tensorflow/issues/20481
            x = x[0] if len(x) == 1 else tuple(x)

            data_set = (
                tf.data.Dataset.from_tensor_slices(x)
                .batch(batch_size)
                .repeat(repeat_count)
            )
            if shuffle:
                data_set = data_set.shuffle(buffer_size, seed=random_seed)
        else:
            data_set = dataset_builder(x)
            steps_per_epoch = data_set.steps_per_epoch
            self.steps_per_epoch = steps_per_epoch

        data_set = iter(data_set)

        if stage == "train":
            self.train_set = data_set
        elif stage == "eval":
            self.eval_set = data_set
        else:
            raise Exception(f"Illegal argument stage={stage}")

        return steps_per_epoch

    def base_forward(self, stage="train"):
        """compute hidden embedding
        Args:
            stage: Which stage of the base forward
        Returns: hidden embedding
        """
        assert (
            self.model_base is not None
        ), "Base model cannot be none, please give model define or load a trained model"

        data_x = None
        self.init_data()
        if stage == "train":
            train_data = next(self.train_set)
            if self.has_y:
                if self.has_s_w:
                    data_x = train_data[:-2]
                    self.train_y = train_data[-2]
                    self.train_sample_weight = train_data[-1]
                else:
                    data_x = train_data[:-1]
                    self.train_y = train_data[-1]
            else:
                data_x = train_data
        elif stage == "eval":
            eval_data = next(self.eval_set)
            if self.has_y:
                if self.has_s_w:
                    data_x = eval_data[:-2]
                    self.eval_y = eval_data[-2]
                    self.eval_sample_weight = eval_data[-1]
                else:
                    data_x = eval_data[:-1]
                    self.eval_y = eval_data[-1]
            else:
                data_x = eval_data
        else:
            raise Exception("invalid stage")

        # Strip tuple of length one, e.g: (x,) -> x
        data_x = data_x[0] if isinstance(data_x, Tuple) and len(data_x) == 1 else data_x

        self.tape = tf.GradientTape()
        with self.tape:
            self.h = self.model_base(data_x)

            # NOTE: For graph neural network, we don't know (node/edge) labels ahead,
            # so the model should return both label and prediction.
            # TODO(@wuxibin): find a better way to extract graph labels.
            if isinstance(self.h, tuple) and len(self.h) == 2:
                y_true, self.h = self.h[0], self.h[1]
                if stage == "train":
                    self.train_y = y_true
                else:
                    self.eval_y = y_true
            elif isinstance(self.h, tuple) and len(self.h) == 3:
                y_true, self.h, self.kwargs = self.h[0], self.h[1], self.h[2]
                if stage == "train":
                    self.train_y = y_true
                else:
                    self.eval_y = y_true

            # Embedding differential privacy
            if self.embedding_dp is not None:
                self.h = self.embedding_dp(self.h)

        return self.h

    def base_backward(self, gradient):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
        """
        with self.tape:
            h = self.fuse_op(self.h, gradient)

        trainable_vars = self.model_base.trainable_variables
        gradients = self.tape.gradient(h, trainable_vars)
        self.model_base.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # clear intermediate results
        self.tape = None
        self.h = None
        self.kwargs = {}

    def get_base_weights(self):
        return self.model_base.get_weights()

    def get_fuse_weights(self):
        return self.model_fuse.get_weights() if self.model_fuse is not None else None

    def init_training(self, callbacks, epochs=1, steps=0, verbose=0):
        if not isinstance(callbacks, callbacks_module.CallbackList):
            self.fuse_callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self.model_fuse,
                verbose=verbose,
                epochs=epochs,
                steps=steps,
                steps_per_epoch=self.steps_per_epoch,
            )
        else:
            raise NotImplementedError

    def get_stop_training(self):
        return self.model_fuse.stop_training

    def on_train_begin(self):
        self.fuse_callbacks.on_train_begin()

    def on_epoch_begin(self, epoch):
        self.fuse_callbacks.on_epoch_begin(epoch)

    def on_train_batch_begin(self, step=None):
        assert step is not None, "Step cannot be none"
        self.fuse_callbacks.on_train_batch_begin(step)

    def on_train_batch_end(self, step=None):
        assert step is not None, "Step cannot be none"
        self.epoch_logs = copy.deepcopy(self.logs)
        self.fuse_callbacks.on_train_batch_end(step, self.logs)

    def on_validation(self, val_logs):
        val_logs = {'val_' + name: val for name, val in val_logs.items()}
        self.epoch_logs.update(val_logs)

    def on_epoch_end(self, epoch):

        self.fuse_callbacks.on_epoch_end(epoch, self.epoch_logs)
        self.training_logs = self.epoch_logs
        return self.epoch_logs

    def on_train_end(self):
        self.fuse_callbacks.on_train_end(logs=self.training_logs)
        return self.model_fuse.history.history

    def set_sample_weight(self, sample_weight, stage="train"):
        if stage == "train":
            self.train_sample_weight = sample_weight
        elif stage == "eval":
            self.eval_sample_weight = sample_weight
        else:
            raise Exception("Illegal Argument")

    def fuse_net(self, *hidden_features) -> Tuple[np.ndarray, np.ndarray]:
        """Fuses the hidden layer and calculates the reverse gradient
        only on the side with the label

        Args:
            hiddens: A list of hidden layers for each party to compute
        Returns:
            gradient Of hiddens
        """
        assert (
            self.model_fuse is not None
        ), "Fuse model cannot be none, please give model define"

        hiddens = [tf.convert_to_tensor(h) for h in hidden_features]

        logs = {}
        with tf.GradientTape(persistent=True) as tape:
            for h in hiddens:
                tape.watch(h)

            # Step 1: forward pass
            y_pred = self.model_fuse(hiddens, training=True, **self.kwargs)

            # Step 2: loss calculation, the loss function is configured in `compile()`.
            loss = self.model_fuse.compiled_loss(
                self.train_y,
                y_pred,
                sample_weight=self.train_sample_weight,
                regularization_losses=self.model_fuse.losses,
            )

        # Step3: compute gradients
        trainable_vars = self.model_fuse.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.model_fuse.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Step4: update metrics
        self.model_fuse.compiled_metrics.update_state(
            self.train_y, y_pred, sample_weight=self.train_sample_weight
        )
        for m in self.model_fuse.metrics:
            logs['train_' + m.name] = m.result().numpy()
        self.logs = logs
        return tape.gradient(loss, hiddens)

    def reset_metrics(self):
        self.model_fuse.compiled_metrics.reset_state()
        self.model_fuse.compiled_loss.reset_state()

    def evaluate(self, *hidden_features):
        assert (
            self.model_fuse is not None
        ), "model cannot be none, please give model define"

        assert self.eval_y is not None, "eval_y cannot be empty"
        hiddens = [tf.convert_to_tensor(h) for h in hidden_features]

        # Step 1: forward pass
        y_pred = self.model_fuse(hiddens, training=False, **self.kwargs)
        # Step 2: update metrics
        self.model_fuse.compiled_metrics.update_state(
            self.eval_y, y_pred, sample_weight=self.eval_sample_weight
        )
        # Step 3: update loss
        self.model_fuse.compiled_loss(
            self.eval_y,
            y_pred,
            sample_weight=self.eval_sample_weight,
            regularization_losses=self.model_fuse.losses,
        )
        result = {}
        for m in self.model_fuse.metrics:
            result[m.name] = m.result().numpy()
        return result

    def metrics(self):
        return self.model_fuse.metrics

    def predict(self, *hidden_features):
        assert (
            self.model_fuse is not None
        ), "Fuse model cannot be none, please give model define"

        hiddens = [tf.convert_to_tensor(h) for h in hidden_features]
        y_pred = self.model_fuse(hiddens)
        return y_pred

    def save_base_model(self, base_model_path: str, save_traces=True):
        assert base_model_path is not None, "model path cannot be empty"
        self.model_base.save(base_model_path, save_traces=save_traces)

    def save_fuse_model(self, fuse_model_path: str, save_traces=True):
        assert fuse_model_path is not None, "model path cannot be empty"
        self.model_fuse.save(fuse_model_path, save_traces=save_traces)

    def load_base_model(self, base_model_path: str, custom_objects=None):
        assert base_model_path is not None, "model path cannot be empty"
        self.model_base = tf.keras.models.load_model(
            base_model_path, custom_objects=custom_objects
        )

    def load_fuse_model(self, fuse_model_path: str, custom_objects=None):
        assert fuse_model_path is not None, "model path cannot be empty"
        self.model_fuse = tf.keras.models.load_model(
            fuse_model_path, custom_objects=custom_objects
        )

    def get_privacy_spent(self, step: int, orders=None):
        """Get accountant of dp mechanism.

        Args:
            step: The current step of model training or prediction.
            orders: An array (or a scalar) of RDP orders.
        """
        privacy_dict = self.dp_strategy.get_privacy_spent(step, orders)
        return privacy_dict


class ModelPartition(object):
    def __init__(self, model_fn, optim_fn, loss_fn, dataloader_fn):
        self.model: SLBaseModule = model_fn()
        self.optimizer: Optimizer = optim_fn(self.model.parameters())
        self.loss: BaseTorchLoss = loss_fn()
        self._dataloader: Dict[str, DataLoader] = {
            k: dl_fn() for k, dl_fn in dataloader_fn.items()
        }
        self._dataiter: Dict[str, Iterator] = {
            k: iter(_dl) for k, _dl in self._dataloader.items()
        }

    def get_one_batch(self, name='train'):
        try:
            x, y = next(self._dataiter[name])
        except StopIteration:
            self._dataiter = iter(self._dataloader[name])
            x, y = next(self._dataiter)
        return x, y

    def forward(
        self, used_name='train', external_input=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if external_input is None:
            external_input = {}
        x, y = self.get_one_batch(used_name)
        y_pred = self.model(x, **external_input)
        return y_pred, y

    def zero_grad(self):
        self.optimizer.zero_grad()

    def backward(self, used_name='train', gradients=None, external_input: Dict = None):
        if gradients is not None:
            self.model.set_gradients(gradients)
        else:
            if external_input is None:
                external_input = {}
            y_pred, y = self.forward(used_name, external_input)
            loss = self.loss(y_pred, y)
            loss.backward()
        return self.model.get_gradients()

    def apply_gradients(self, gradients=None):
        if gradients is not None:
            self.model.set_gradients(gradients)
        self.optim_step()

    def optim_step(self):
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def call_model_fn(self, fn_name, *args, **kwargs):
        # TODO: a temporary utils
        return getattr(self.model, fn_name)(*args, **kwargs)


@proxy(PYUObject)
class PYUSLTFModel(SLBaseTFModel):
    pass


@proxy(ft.PYUObject)
class PYUModel(ModelPartition):
    pass
