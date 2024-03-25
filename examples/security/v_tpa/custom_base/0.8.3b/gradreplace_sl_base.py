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
        compressor: Compressor,
        random_seed: int = None,
        **kwargs,
    ):
        super().__init__(
            builder_base, builder_fuse, dp_strategy, compressor, random_seed, **kwargs
        )

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

    @tf.function
    def _base_forward_internal(self, data_x):
        h = self.model_base(data_x)

        # Embedding differential privacy
        if self.embedding_dp is not None:
            if isinstance(h, List):
                h = [self.embedding_dp(hi) for hi in h]
            else:
                h = self.embedding_dp(h)
        return h

    def base_forward(self, stage="train", compress: bool = False) -> ForwardData:
        """compute hidden embedding
        Args:
            stage: Which stage of the base forward
            compress: Whether to compress cross device data.
        Returns: hidden embedding
        """

        assert (
            self.model_base is not None
        ), "Base model cannot be none, please give model define or load a trained model"

        data_x = None
        self.init_data()
        if stage == "train":
            train_data = next(self.train_set)
            if self.train_has_y:
                if self.train_has_s_w:
                    data_x = train_data[:-2]
                    train_y = train_data[-2]
                    self.train_sample_weight = train_data[-1]
                else:
                    data_x = train_data[:-1]
                    train_y = train_data[-1]
                # Label differential privacy
                if self.label_dp is not None:
                    dp_train_y = self.label_dp(train_y.numpy())
                    self.train_y = tf.convert_to_tensor(dp_train_y)
                else:
                    self.train_y = train_y
            else:
                data_x = train_data
        elif stage == "eval":
            eval_data = next(self.eval_set)
            if self.eval_has_y:
                if self.eval_has_s_w:
                    data_x = eval_data[:-2]
                    eval_y = eval_data[-2]
                    self.eval_sample_weight = eval_data[-1]
                else:
                    data_x = eval_data[:-1]
                    eval_y = eval_data[-1]
                # Label differential privacy
                if self.label_dp is not None:
                    dp_eval_y = self.label_dp(eval_y.numpy())
                    self.eval_y = tf.convert_to_tensor(dp_eval_y)
                else:
                    self.eval_y = eval_y
            else:
                data_x = eval_data
        else:
            raise Exception("invalid stage")

        # Strip tuple of length one, e.g: (x,) -> x
        # modify: gradient replacement needs features and indexes
        assert len(data_x) >= 2
        data_indexes = data_x[-1]
        data_x = (
            data_x[0]
            if isinstance(data_x[:-1], Tuple) and len(data_x[:-1]) == 1
            else data_x[:-1]
        )

        if stage == "train":
            # replacement attack
            data_x = self.forward_replace(data_indexes, data_x)

        self.tape = tf.GradientTape(persistent=True)
        with self.tape:
            self.h = self._base_forward_internal(data_x)

        self.data_x = data_x

        forward_data = ForwardData()
        if len(self.model_base.losses) > 0:
            forward_data.losses = tf.add_n(self.model_base.losses)
        # TODO: only vaild on no server mode, refactor when use agglayer or server mode.
        # no need to compress data on model_fuse side
        if compress and not self.model_fuse:
            if self.compressor:
                forward_data.hidden = self.compressor.compress(self.h.numpy())
            else:
                raise Exception(
                    "can not find compressor when compress data in base_forward"
                )
        else:
            forward_data.hidden = self.h
        return forward_data

    @tf.function
    def _base_backward_internal(self, gradients, trainable_vars):
        self.model_base.optimizer.apply_gradients(zip(gradients, trainable_vars))

    def base_backward(self, gradient, compress: bool = False):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
            compress: Whether to decompress gradient.
        """
        return_hiddens = []

        # TODO: only vaild on no server mode, refactor when use agglayer or server mode.
        # no need to decompress data on model_fuse side
        if compress and not self.model_fuse:
            if self.compressor:
                gradient = self.compressor.decompress(gradient)
            else:
                raise Exception(
                    "can not find compressor when decompress data in base_backward"
                )

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
