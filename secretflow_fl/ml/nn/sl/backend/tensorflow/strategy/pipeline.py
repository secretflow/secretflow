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
from typing import Callable

import tensorflow as tf

from secretflow_fl.ml.nn.sl.backend.tensorflow.sl_base import ListType, SLBaseTFModel
from secretflow_fl.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow_fl.security.privacy import DPStrategy


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
        self.base_loss_list = []
        self._pre_train_y = []

    def reset_data_iter(self, stage="train"):
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
        self.base_loss_list = []
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
            if isinstance(self._data_x, ListType) and len(self._data_x) == 1
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
            self.base_loss_list.append(self._base_losses)

    def base_backward(self):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
        """
        # TODO: only vaild on no server mode, refactor when use agglayer or server mode.
        # no need to decompress data on model_fuse side
        pre_tape = self.base_tape.pop(0)
        hiddens = self.hidden_list.pop(0)
        base_losses = self.base_loss_list.pop(0)
        return_hiddens = self._base_backward_hidden_internal(
            pre_tape, hiddens, self._gradient, base_losses
        )

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

        self.tape = None
        self._h = None
        self._base_losses = None
        self.kwargs = {}

    def _fuse_net_train(self, hiddens, losses=[]):
        train_y = self._pre_train_y.pop(0)
        return self._fuse_net_internal(
            hiddens,
            losses,
            train_y,
            self.train_sample_weight,
        )


@register_strategy(strategy_name="pipeline", backend="tensorflow")
class PYUPipelineTFModel(PipelineTFModel):
    pass
