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
from secretflow_fl.ml.nn.metrics import AUC, Mean, Precision, Recall
from secretflow_fl.ml.nn.sl.backend.tensorflow.utils import ForwardData
from secretflow_fl.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow_fl.security.privacy import DPStrategy
from secretflow_fl.utils.compressor import Compressor, SparseCompressor

from .sl_base import SLBaseTFModel


@register_strategy(strategy_name="replay_split_nn", backend="tensorflow")
@proxy(PYUObject)
class Replay_PYUSLTFModel(SLBaseTFModel):
    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        builder_fuse: Callable[[], tf.keras.Model],
        dp_strategy: DPStrategy,
        random_seed: int = None,
        **kwargs,
    ):
        super().__init__(builder_base, builder_fuse, dp_strategy, random_seed, **kwargs)

        self.attack_args = kwargs.get("attack_args", None)
        self.train_target_indexes = self.attack_args["train_target_indexes"]
        self.valid_poisoning_indexes = self.attack_args["valid_poisoning_indexes"]
        self.train_target_embeddings = None
        self.target_len = len(self.attack_args["train_target_indexes"])
        self.record_counter = 0

    def forward_record(self, data_indexes, embeddings):
        # find out target samples in a batch
        target_set = np.intersect1d(data_indexes, self.train_target_indexes)
        target_offsets = np.where(data_indexes == target_set[:, None])[-1]

        tlen = len(target_offsets)

        # record target embeddings
        if tlen > 0:
            embeddings_np = embeddings.numpy()

            tshape = (tlen,) + embeddings_np.shape[1:]
            batch_embeddings = embeddings_np[target_offsets] + np.random.randn(*tshape)
            batch_embeddings = embeddings_np[target_offsets]
            embeddings_np[target_offsets] = batch_embeddings

            if self.train_target_embeddings is None:
                self.train_target_embeddings = np.zeros(
                    (self.target_len,) + embeddings.shape[1:]
                )

            self.train_target_embeddings[
                self.record_counter : self.record_counter + tlen
            ] = batch_embeddings
            self.record_counter += tlen
            if self.record_counter >= self.target_len:
                self.record_counter -= self.target_len

            embeddings = tf.convert_to_tensor(embeddings_np)
        return embeddings

    def forward_replay(self, data_indexes, embeddings):
        # find out poison samples in a batch
        poison_set = np.intersect1d(data_indexes, self.valid_poisoning_indexes)
        poison_offsets = np.where(data_indexes == poison_set[:, None])[-1]
        plen = len(poison_offsets)

        # replay target embeddings
        if plen > 0 and len(self.train_target_embeddings) > 0:
            embeddings_np = embeddings.numpy()
            replay_keys = np.random.choice(
                np.arange(self.target_len), (plen,), replace=True
            )
            embeddings_np[poison_offsets] = self.train_target_embeddings[replay_keys]
            embeddings = tf.convert_to_tensor(embeddings_np)

        return embeddings

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

        self.tape = tf.GradientTape(persistent=True)
        with self.tape:
            self.h = self._base_forward_internal(data_x, training=training)
        self.data_x = data_x

        # modify:
        # replay attack
        if stage == "train":
            attack_h = self.forward_record(data_indexes, self.h)
        elif stage == "eval":
            attack_h = self.forward_replay(data_indexes, self.h)

        forward_data = ForwardData()
        if len(self.model_base.losses) > 0:
            forward_data.losses = tf.add_n(self.model_base.losses)
        # The compressor can only recognize np type but not tensor.
        forward_data.hidden = self.h.numpy() if tf.is_tensor(self.h) else self.h
        return forward_data
