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

from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.sl.backend.tensorflow.sl_base import SLBaseTFModel
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.security.privacy import DPStrategy
from secretflow.ml.nn.sl.backend.tensorflow.utils import ForwardData


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
        self.h = []
        self.pre_train_y = []

    def base_forward(
        self, stage="train", step=0, compress: bool = False
    ) -> ForwardData:
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
        training = True
        self.init_data()
        if step == 0:
            self._reset_data_iter(stage=stage)

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
                self.pre_train_y.append(self.train_y)
            else:
                data_x = train_data
        elif stage == "eval":
            training = False
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
        data_x = data_x[0] if isinstance(data_x, Tuple) and len(data_x) == 1 else data_x

        tape = tf.GradientTape(persistent=True)
        if stage == "train":
            self.base_tape.append(tape)
            self.trainable_vars.append(self.model_base.trainable_variables)
        with tape:
            h = self._base_forward_internal(
                data_x,
                training=training,
            )
        if stage == "train":
            self.h.append(h)

        self.data_x = data_x

        forward_data = ForwardData()
        if len(self.model_base.losses) > 0:
            forward_data.losses = tf.add_n(self.model_base.losses)
        # TODO: only vaild on no server mode, refactor when use agglayer or server mode.
        # no need to compress data on model_fuse side
        forward_data.hidden = h
        return forward_data

    def base_backward(self, gradient):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
        """
        return_hiddens = []

        # TODO: only vaild on no server mode, refactor when use agglayer or server mode.
        # no need to decompress data on model_fuse side
        pre_tape = self.base_tape.pop(0)
        with pre_tape:
            h = self.h.pop(0)
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
        train_y = self.pre_train_y.pop(0)
        return self._fuse_net_internal(
            hiddens,
            losses,
            train_y,
            self.train_sample_weight,
        )

    def on_epoch_end(self, epoch):
        # clean pipeline
        self.trainable_vars = []
        self.base_tape = []
        self.fuse_tape = []
        self.h = []
        self.pre_train_y = []

        if self.fuse_callbacks:
            self.fuse_callbacks.on_epoch_end(epoch, self.epoch_logs)
        self.training_logs = self.epoch_logs
        return self.epoch_logs


@register_strategy(strategy_name='pipeline', backend='tensorflow')
@proxy(PYUObject)
class PYUPipelineTFModel(PipelineTFModel):
    pass
