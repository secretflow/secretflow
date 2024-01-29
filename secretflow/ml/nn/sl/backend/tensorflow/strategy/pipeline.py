#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# Copyright 2023 Ant Group Co., Ltd.
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

"""pipeline split learning strategy
"""
from typing import Callable, Tuple

import tensorflow as tf

from secretflow.ml.nn.sl.backend.tensorflow.sl_base import SLBaseTFModel
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.security.privacy import DPStrategy


class PipelineTFModel(SLBaseTFModel):
    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        builder_fuse: Callable[[], tf.keras.Model],
        dp_strategy: DPStrategy,
        random_seed: int = None,
        pipeline_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            builder_base,
            builder_fuse,
            dp_strategy,
            random_seed,
            **kwargs,
        )
        self.pipeline_size = pipeline_size

        self.trainable_vars = []
        self.base_tape = []
        self.fuse_tape = []
        self._h = []
        self.hidden_list = []
        self._pre_train_y = []

    def reset_data_iter(self, stage='train'):
        if stage == "train":
            self.train_set = iter(self.train_dataset)
        elif stage == "eval":
            self.eval_set = iter(self.eval_dataset)
        # reset some status
        self.trainable_vars = []
        self.base_tape = []
        self.fuse_tape = []
        self._h = []
        self.hidden_list = []
        self._pre_train_y = []

    def base_forward(self, stage="train", step=0, compress: bool = False):
        """compute hidden embedding
        Args:
            stage: Which stage of the base forward
            compress: Whether to compress cross device data.
        Returns: hidden embedding
        """
        assert (
            self.model_base is not None
        ), "Base model cannot be none, please give model define or load a trained model"

        # Strip tuple of length one, e.g: (x,) -> x
        self._data_x = (
            self._data_x[0]
            if isinstance(self._data_x, Tuple) and len(self._data_x) == 1
            else self._data_x
        )

        tape = tf.GradientTape(persistent=True)
        if stage == "train":
            self.base_tape.append(tape)
            self.trainable_vars.append(self.model_base.trainable_variables)
        with tape:
            self._h = self._base_forward_internal(
                self._data_x,
                training=self._training,
            )
        if stage == "train":
            self.hidden_list.append(self._h)

    def base_backward(self):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
        """
        return_hiddens = []
        gradient = self._gradient
        # TODO: only vaild on no server mode, refactor when use agglayer or server mode.
        # no need to decompress data on model_fuse side
        pre_tape = self.base_tape.pop(0)
        with pre_tape:
            h = self.hidden_list.pop(0)
            if len(gradient) == len(h):
                for i in range(len(gradient)):
                    return_hiddens.append(self.fuse_op(h[i], gradient[i]))
            else:
                gradient = gradient[0]
                return_hiddens.append(self.fuse_op(h, gradient))
            return_hiddens.append(self.model_base.losses)

        now_value = [var.read_value() for var in self.model_base.trainable_variables]
        pre_value = self.trainable_vars.pop(0)
        for idx, var in enumerate(self.model_base.trainable_variables):
            var.assign(pre_value[idx].read_value())
        gradients = pre_tape.gradient(
            return_hiddens, self.model_base.trainable_variables
        )
        for idx, var in enumerate(self.model_base.trainable_variables):
            var.assign(now_value[idx])
        self._base_backward_internal(gradients, self.model_base.trainable_variables)

        self.kwargs = {}

    def _fuse_net_train(self, hiddens, losses=[]):
        train_y = self._pre_train_y.pop(0)
        return self._fuse_net_internal(
            hiddens,
            losses,
            train_y,
            self.train_sample_weight,
        )


@register_strategy(strategy_name='pipeline', backend='tensorflow')
class PYUPipelineTFModel(PipelineTFModel):
    pass
