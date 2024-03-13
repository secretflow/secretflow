#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import pdb
from abc import ABC, abstractmethod
from pathlib import Path
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
from secretflow.ml.nn.metrics import AUC, Mean, Precision, Recall
from secretflow.ml.nn.sl.backend.tensorflow.utils import ForwardData
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.utils.compressor import Compressor, SparseCompressor

# from secretflow.ml.nn.sl.backend.tensorflow.sl_base import SLBaseTFModel
from .sl_base import SLBaseTFModel


@register_strategy(strategy_name="gradreplace_split_nn", backend="tensorflow")
@proxy(PYUObject)
class GRADReplace_PYUSLTFModel(SLBaseTFModel):
    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        builder_fuse: Callable[[], tf.keras.Model],
        dp_strategy: DPStrategy,
        random_seed: int = None,
        **kwargs,
    ):
        super().__init__(builder_base, builder_fuse, dp_strategy, random_seed, **kwargs)
        self.attack_args = kwargs.get("attack_args", {})
        self.train_poisoning_indexes = self.attack_args["train_poisoning_indexes"]
        self.train_target_indexes = self.attack_args["train_target_indexes"]
        self.train_features = self.attack_args["train_features"]
        self.valid_poisoning_indexes = self.attack_args["valid_poisoning_indexes"]
        self.valid_target_indexes = self.attack_args["valid_target_indexes"]
        self.blurred = self.attack_args["blurred"]
        self.gamma = self.attack_args["gamma"]

    def forward_replace(self, data_indexes, inputs):
        # find out the poison and target samples in a batch
        self.poisoning_set = np.intersect1d(data_indexes, self.train_poisoning_indexes)
        self.target_set = np.intersect1d(data_indexes, self.train_target_indexes)

        self.target_offsets = np.where(data_indexes == self.target_set[:, None])[-1]
        self.poison_offsets = np.where(data_indexes == self.poisoning_set[:, None])[-1]

        plen = len(self.poison_offsets)
        tlen = len(self.target_offsets)

        if plen > 0 or tlen > 0:
            inputs_np = inputs.numpy()
            if tlen > 0:
                choices = np.random.choice(
                    self.train_poisoning_indexes, (tlen,), replace=True
                )
                inputs_np[self.target_offsets] = self.train_features[choices]

            if self.blurred and plen > 0:
                rnd_shape = (plen,) + inputs.shape[1:]
                inputs_np[self.poison_offsets] = np.random.randn(*rnd_shape)

            inputs = tf.convert_to_tensor(inputs_np)
        return inputs

    def gradient_replace(self, inputs):
        if len(self.target_offsets) > 0:
            inputs_np = inputs.numpy()
            inputs_np[self.target_offsets] = self.gamma * inputs_np[self.target_offsets]

            inputs = tf.convert_to_tensor(inputs_np)
        return inputs

    def base_forward(self, stage="train") -> ForwardData:
        """compute hidden embedding
        Args:
            stage: Which stage of the base forward
        Returns: hidden embedding
        """
        data_x = None
        self.init_data()
        training = True
        if stage == "train":
            train_data = next(self.train_set)

            self.train_x, self.train_y, self.train_sample_weight = self.unpack_dataset(
                train_data, self.train_has_x, self.train_has_y, self.train_has_s_w
            )
            data_x = self.train_x
        elif stage == "eval":
            training = False
            eval_data = next(self.eval_set)
            self.eval_x, self.eval_y, self.eval_sample_weight = self.unpack_dataset(
                eval_data, self.eval_has_x, self.eval_has_y, self.eval_has_s_w
            )
            data_x = self.eval_x
        else:
            raise Exception("invalid stage")

        # model_base is none equal to x is none
        if not self.model_base:
            return None

        # Strip tuple of length one, e.g: (x,) -> x
        # modify: gradient replacement needs features and indexes
        assert len(data_x) >= 2
        data_indexes = data_x[-1]
        data_x = (
            data_x[0]
            if isinstance(data_x[:-1], Tuple) and len(data_x[:-1]) == 1
            else data_x[:-1]
        )

        # modify:
        # replacement attack
        if stage == "train":
            data_x = self.forward_replace(data_indexes, data_x)

        self.tape = tf.GradientTape(persistent=True)
        with self.tape:
            self.h = self._base_forward_internal(data_x, training=training)
        self.data_x = data_x

        forward_data = ForwardData()
        if len(self.model_base.losses) > 0:
            forward_data.losses = tf.add_n(self.model_base.losses)
        # The compressor can only recognize np type but not tensor.
        forward_data.hidden = self.h.numpy() if tf.is_tensor(self.h) else self.h
        return forward_data

    def base_backward(self, gradient):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
        """

        return_hiddens = []

        # modify: replacement attack
        if isinstance(gradient, List):
            new_gradient = []
            for grads in gradient:
                grads = self.gradient_replace(grads)
                new_gradient.append(grads)
            gradient = new_gradient
        else:
            gradient = self.gradient_replace(gradient)

        with self.tape:
            if len(gradient) == len(self.h):
                for i in range(len(gradient)):
                    return_hiddens.append(self.fuse_op(self.h[i], gradient[i]))
            else:
                gradient = gradient[0]
                return_hiddens.append(self.fuse_op(self.h, gradient))
            # add model.losses into graph
            return_hiddens.append(self.model_base.losses)

        trainable_vars = self.model_base.trainable_variables
        gradients = self.tape.gradient(return_hiddens, trainable_vars)

        self._base_backward_internal(gradients, trainable_vars)

        # clear intermediate results
        self.tape = None
        self.h = None
        self.kwargs = {}
