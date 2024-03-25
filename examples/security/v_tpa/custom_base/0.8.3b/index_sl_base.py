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


class IndexSLBaseTFModel(SLBaseTFModel):
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

        self.tape = tf.GradientTape(persistent=True)
        with self.tape:
            self.h = self._base_forward_internal(
                data_x,
            )
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


@register_strategy(strategy_name="index_split_nn", backend="tensorflow")
@proxy(PYUObject)
class IndexPYUSLTFModel(IndexSLBaseTFModel):
    pass
