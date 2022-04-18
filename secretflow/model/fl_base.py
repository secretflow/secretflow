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

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Union, Optional, Callable, Tuple

import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torch.nn.modules.loss import _Loss as BaseTorchLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import secretflow.device as ft
from secretflow.device import PYUObject
from secretflow.device import proxy

# 抽象model类


class BaseModel(ABC):
    def __init__(self, builder_base: Callable, builder_fuse: Callable = None):
        self.model_base = builder_base()
        self.model_fuse = builder_fuse() if builder_fuse is not None else None

    @abstractmethod
    def build_dataset(self, x: np.ndarray, y: Optional[np.ndarray] = None, batch_size=32, buffer_size=128,
                      repeat_count=1):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def evaluate(self, x, y, batch_size=None, verbose=1, sample_weight=None, steps=None):
        pass


class BaseTFModel(BaseModel):
    def __init__(self, builder_base: Callable[[], tf.keras.Model]):
        super().__init__(builder_base)
        self.model = builder_base()
        self.train_set = None
        self.has_s_w = False
    # TODO: 兼容HDataFrame HNDarray

    def build_dataset(self, x: np.ndarray,
                      y: Optional[np.ndarray] = None,
                      s_w: Optional[np.ndarray] = None,
                      batch_size=32, buffer_size=128, shuffle=False,
                      repeat_count=1):
        """构建tf.data.Dataset

        Args:
            x: 训练样本
            y: 样本标签，仅当训练时需要
            batch_size: 批次大小
            buffer_size: shuffle的缓存大小
            repeat_count: 重复次数
        """
        data_set = None
        self.has_s_w = False
        # construct train_set
        if x is None or len(x.shape) == 0:
            raise Exception("Data 'x' cannot be None")
        if y is None or len(y.shape) == 0:
            raise Exception("Data 'y' cannot be None")
        if s_w is not None:
            self.has_s_w = True
            data_set = tf.data.Dataset.from_tensor_slices((x, y.astype(
                np.float64), s_w.astype(np.float64))).batch(batch_size).repeat(repeat_count)
        else:
            data_set = tf.data.Dataset.from_tensor_slices((x, y.astype(
                np.float64))).batch(batch_size).repeat(repeat_count)

        if shuffle:
            data_set = data_set.shuffle(buffer_size)
        self.train_set = iter(data_set)

    # TODO compute_gradients():

    def get_weights(self):
        return self.model.get_weights()

    def get_metric_name(self):
        return [m.name for m in self.model.metrics]

    def get_metrics(self):
        return [m.result().numpy() for m in self.model.metrics]

    def evaluate(self, x, y, batch_size=None, verbose=0, sample_weight=None, steps=None):
        local_metric = self.model.evaluate(
            x, y, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight, steps=steps)
        return np.array(local_metric)

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                workers=1, use_multiprocessing=False):
        local_pred = self.model.predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks,
                                        max_queue_size=max_queue_size,
                                        workers=workers, use_multiprocessing=use_multiprocessing)
        return local_pred

    def train_step(self, weights, cur_steps, train_steps) -> Tuple[np.ndarray, int]:
        """根据参数服务器最新参数，进行本地训练

        Args:
            weights: 全局最新参数
            cur_steps: 当前训练起始步数
            train_steps: 本地训练步数

        Returns:
            本地训练后的最新参数
        """
        self.model.set_weights(weights)
        num_sample = 0
        for _ in range(train_steps):
            if self.has_s_w:
                x, y, s_w = next(self.train_set)
            else:
                x, y = next(self.train_set)
                s_w = None
            num_sample += x.shape[0]
            with tf.GradientTape() as tape:
                # Step 1: forward pass
                y_pred = self.model(x, training=True)
                # Step 2: loss calculation, the loss function is configured in `compile()`.
                loss = self.model.compiled_loss(
                    y, y_pred,
                    regularization_losses=self.model.losses,
                    sample_weight=s_w,
                )
            # Step 3: back propagation
            trainable_vars = self.model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.model.optimizer.apply_gradients(
                zip(gradients, trainable_vars))
            # Step4: update metrics
            self.model.compiled_metrics.update_state(y, y_pred)
        return self.model.get_weights(), num_sample

    def save_model(self, model_path: str):
        assert model_path is not None, "model path cannot be empty"
        self.model.save(model_path)

    def load_model(self, model_path: str):
        assert model_path is not None, "model path cannot be empty"
        self.model = tf.keras.models.load_model(model_path)


class BaseModule(ABC, nn.Module):
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

    def set_gradients(self, gradients: List[Union[torch.Tensor, np.ndarray]],
                      parameters: Optional[List[torch.Tensor]] = None):
        if parameters is None:
            parameters = self.parameters()
        for g, p in zip(gradients, parameters):
            if g is not None:
                p.grad = torch.from_numpy(g.copy())


class ModelPartition(object):
    def __init__(self, model_fn, optim_fn, loss_fn, dataloader_fn):
        self.model: BaseModule = model_fn()
        self.optimizer: Optimizer = optim_fn(self.model.parameters())
        self.loss: BaseTorchLoss = loss_fn()
        self._dataloader: Dict[str, DataLoader] = {
            k: dl_fn() for k, dl_fn in dataloader_fn.items()}
        self._dataiter: Dict[str, Iterator] = {
            k: iter(_dl) for k, _dl in self._dataloader.items()}

    def get_one_batch(self, name='train'):
        try:
            x, y = next(self._dataiter[name])
        except StopIteration:
            self._dataiter = iter(self._dataloader[name])
            x, y = next(self._dataiter)
        return x, y

    def forward(self, used_name='train', external_input=None) -> (torch.Tensor, torch.Tensor):
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
class PYUTFModel(BaseTFModel):
    pass


@proxy(ft.PYUObject)
class PYUModel(ModelPartition):
    pass
