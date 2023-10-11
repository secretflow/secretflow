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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import callbacks as callbacks_module

from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.metrics import AUC, Mean, Precision, Recall
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.utils.communicate import ForwardData

from .sl_base import SLBaseTFModel


class IndexSLBaseTFModel(SLBaseTFModel):
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
        data_x = data_x[0] if isinstance(data_x[:-1], Tuple) and len(data_x[:-1]) == 1 else data_x[:-1]
        # data_x = data_x[0] if isinstance(data_x, Tuple) and len(data_x) == 1 else data_x

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

@register_strategy(strategy_name='index_split_nn', backend='tensorflow')
@proxy(PYUObject)
class IndexPYUSLTFModel(IndexSLBaseTFModel):
    pass
