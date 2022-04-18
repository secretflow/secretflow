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

#!/usr/bin/env python3
# *_* coding: utf-8 *_*

""" 抽象model类
"""
from ctypes import ArgumentError
from typing import Dict, Iterator, List, Union, Optional, Callable, Tuple

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torch.nn.modules.loss import _Loss as BaseTorchLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from secretflow.device import PYUObject, proxy
import secretflow.device as ft


class SLBaseModel(ABC):
    def __init__(self, builder_base: Callable, builder_fuse: Callable = None):
        self.model_base = builder_base()
        self.model_fuse = builder_fuse() if builder_fuse is not None else None

    @abstractmethod
    def build_dataset(self, x: np.ndarray, y: Optional[np.ndarray] = None, s_w: Optional[np.ndarray] = None, batch_size=32, buffer_size=128,
                      repeat_count=1):
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

    def set_gradients(self, gradients: List[Union[torch.Tensor, np.ndarray]],
                      parameters: Optional[List[torch.Tensor]] = None):
        if parameters is None:
            parameters = self.parameters()
        for g, p in zip(gradients, parameters):
            if g is not None:
                p.grad = torch.from_numpy(g.copy())


class SLBaseTFModel(SLBaseModel):
    def __init__(self, builder_base: Callable[[], tf.keras.Model], builder_fuse: Callable[[], tf.keras.Model]):
        super().__init__(builder_base)
        self.model_base = builder_base()
        self.model_fuse = builder_fuse() if builder_fuse is not None else None

        self.train_set = None
        self.eval_set = None
        self.valid_set = None
        self.tape = None
        self.h = None
        self.train_x, self.train_y = None, None
        self.eval_x, self.eval_y = None, None
        self.has_y = False
        self.has_s_w = False
        self.train_sample_weight = None
        self.eval_sample_weight = None

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

    def build_dataset(self, *x: List[np.ndarray], y: Optional[np.ndarray] = None,
                      s_w: Optional[np.ndarray] = None,
                      batch_size=32, buffer_size=128, shuffle=False,
                      repeat_count=1, stage="train", random_seed=1234):
        """构建tf.data.Dataset

        Args:
            x: 训练样本
            y: 样本标签，仅当训练时需要
            batch_size: 批次大小
            buffer_size: shuffle的缓存大小
            repeat_count: 重复次数
        """
        assert len(
            x) > 0 or x[0] is not None, "X can not be None, please check"
        x = [xi.astype(np.float64) for xi in x]
        self.has_y = False
        self.has_s_w = False
        if y is not None and len(y.shape) > 0:
            self.has_y = True
            x.append(y.astype(np.float64))
            if s_w is not None and len(s_w.shape) > 0:
                self.has_s_w = True
                x.append(s_w.astype(np.float64))
        x = tuple(x)

        data_set = tf.data.Dataset.from_tensor_slices(
            x).batch(batch_size).repeat(repeat_count)

        if shuffle:
            data_set = data_set.shuffle(buffer_size, seed=random_seed)

        if stage == "train":
            self.train_set = iter(data_set)
        elif stage == "eval":
            self.eval_set = iter(data_set)
        else:
            raise ArgumentError(f"Illegal argument stage={stage}")

    # TODO compute_gradients():

    def base_forward(self, stage="train", step=0):
        """计算前向隐层

        Returns: 用于融合的隐层
        """
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

        self.tape = tf.GradientTape()
        with self.tape:
            self.h = self.model_base(data_x)
        return self.h

    def base_backward(self, gradient):
        """反向梯度更新

        Args:
            gradient: 融合隐层的反向梯度
        """
        with self.tape:
            h = self.fuse_op(self.h, gradient)

        trainable_vars = self.model_base.trainable_variables
        gradients = self.tape.gradient(h, trainable_vars)
        self.model_base.optimizer.apply_gradients(
            zip(gradients, trainable_vars))

        # clear intermediate results
        self.tape = None
        self.h = None

    def get_base_weights(self):
        return self.model_base.get_weights()

    def get_fuse_weights(self):
        return self.model_fuse.get_weights() if self.model_fuse is not None else None

    def set_sample_weight(self, sample_weight, stage="train"):
        if stage == "train":
            self.train_sample_weight = sample_weight
        elif stage == "eval":
            self.eval_sample_weight = sample_weight
        else:
            raise Exception("Illegal Argument")

    def fuse_net(self, *hidden_features) -> (np.ndarray, np.ndarray):
        """融合隐层，计算反向梯度，仅在有label的一方计算

        Args:
            hiddens: 各方计算的隐层列表
            output_y: 是否将y_pred输出
        Returns:
            hiddens的反向梯度
        """
        hiddens = [tf.convert_to_tensor(h) for h in hidden_features]

        with tf.GradientTape(persistent=True) as tape:
            for h in hiddens:
                tape.watch(h)

            # Step 1: forward pass
            y_pred = self.model_fuse(hiddens)

            # Step 2: loss calculation, the loss function is configured in `compile()`.
            loss = self.model_fuse.compiled_loss(
                self.train_y, y_pred,
                sample_weight=self.train_sample_weight,
                regularization_losses=self.model_fuse.losses,
            )

        # Step3: compute gradients
        trainable_vars = self.model_fuse.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.model_fuse.optimizer.apply_gradients(
            zip(gradients, trainable_vars))

        # Step4: update metrics
        self.model_fuse.compiled_metrics.update_state(self.train_y, y_pred)

        return tape.gradient(loss, hiddens)

    def evaluate(self, *hidden_features):
        self.model_fuse.compiled_metrics.reset_state()
        self.model_fuse.compiled_loss.reset_state()
        assert self.eval_y is not None, "eval_y cannot be empty"
        hiddens = [tf.convert_to_tensor(h) for h in hidden_features]

        # Step 1: forward pass
        y_pred = self.model_fuse(hiddens)
        # Step 2: update metrics
        self.model_fuse.compiled_metrics.update_state(self.eval_y, y_pred)
        # Step 3: update loss
        self.model_fuse.compiled_loss(
            self.eval_y, y_pred, sample_weight=self.eval_sample_weight)
        result = {}
        for m in self.model_fuse.metrics:
            result[m.name] = m.result().numpy()
        return result

    def predict(self, *hidden_features):
        hiddens = [tf.convert_to_tensor(h) for h in hidden_features]
        y_pred = self.model_fuse(hiddens)
        return y_pred

    def save_base_model(self, base_model_path: str):
        assert base_model_path is not None, "model path cannot be empty"
        self.model_base.save(base_model_path)

    def save_fuse_model(self, fuse_model_path: str):
        assert fuse_model_path is not None, "model path cannot be empty"
        self.model_fuse.save(fuse_model_path)

    def load_base_model(self, base_model_path: str):
        assert base_model_path is not None, "model path cannot be empty"
        self.model_base = tf.keras.models.load_model(base_model_path)

    def load_fuse_model(self, fuse_model_path: str):
        assert fuse_model_path is not None, "model path cannot be empty"
        self.model_fuse = tf.keras.models.load_model(fuse_model_path)


class ModelPartition(object):
    def __init__(self, model_fn, optim_fn, loss_fn, dataloader_fn):
        self.model: SLBaseModule = model_fn()
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
class PYUSLTFModel(SLBaseTFModel):
    pass


@proxy(ft.PYUObject)
class PYUModel(ModelPartition):
    pass
