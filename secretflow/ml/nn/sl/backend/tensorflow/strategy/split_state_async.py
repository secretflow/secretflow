#!/usr/bin/env python3
# *_* coding: utf-8 *_*

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


""" Stateful async split learning strategy
Reference:
    [1] Chen, X., Li, J., & Chakrabarti, C. Communication and computation reduction for split learning using asynchronous training[C]. arXiv preprint arXiv:2107.09786, 2021.(https://arxiv.org/abs/2107.09786)
"""

from typing import Callable

import tensorflow as tf

from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.sl.backend.tensorflow.sl_base import SLBaseTFModel
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.security.privacy import DPStrategy


class SLStateAsyncTFModel(SLBaseTFModel):
    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        builder_fuse: Callable[[], tf.keras.Model],
        dp_strategy: DPStrategy,
        loss_thres: float = 0.01,
        split_steps: int = 1,
        max_fuse_local_steps: int = 1,
        random_seed: int = None,
        **kwargs,
    ):
        super().__init__(
            builder_base,
            builder_fuse,
            dp_strategy,
            random_seed,
            **kwargs,
        )
        assert (
            max_fuse_local_steps > 0
        ), f'state async max_fuse_local_steps should greater than 0'
        self.loss_thres = loss_thres
        self.split_steps = split_steps
        self.max_fuse_local_steps = max_fuse_local_steps
        # SplitAT state
        self.count = 0
        self.total_loss = 0
        self.last_update_loss = 0
        self.state = 'A'

    def _fuse_net_train(self, hiddens, losses=[]):
        cnt = 0
        while cnt <= self.max_fuse_local_steps:
            cnt += 1
            gradient, state = self._fuse_net_internal(
                hiddens,
                losses,
                self.train_y,
                self.train_sample_weight,
            )
            if state != 'C':
                break
        if state != 'A':
            self.skip_gradient = True
        else:
            self.skip_gradient = False
        return gradient

    def get_skip_gradient(self):
        return self.skip_gradient

    def _fuse_net_internal(self, hiddens, losses, train_y, train_sample_weight):
        with tf.GradientTape(persistent=True) as tape:
            for h in hiddens:
                tape.watch(h)

            # Step 1: forward pass
            y_pred = self.model_fuse(hiddens, training=True, **self.kwargs)
            # Step 2: loss calculation, the loss function is configured in `compile()`.
            loss = self.model_fuse.compiled_loss(
                train_y,
                y_pred,
                sample_weight=train_sample_weight,
                regularization_losses=self.model_fuse.losses + losses,
            )

        # Step3: compute gradients
        trainable_vars = self.model_fuse.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.model_fuse.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Step4: update metrics
        self.model_fuse.compiled_metrics.update_state(
            train_y, y_pred, sample_weight=train_sample_weight
        )

        # check loss
        self.total_loss += loss
        self.count += 1
        # Here we refer to the definition of the state in the paper
        # *Communication and Computation Reduction for Split Learning using Asynchronous Training*
        # | State | Hidden         | Gradient       |
        # |-------|----------------|----------------|
        # | A     | client->server | server->client |
        # | B     | client->server | None           |
        # | C     | None           | None           |
        if self.count >= self.split_steps:
            # update state
            avg_loss = self.total_loss / self.count
            delta = abs(self.last_update_loss - avg_loss)
            if delta >= self.loss_thres:
                self.state = 'A'
            else:
                if self.state == 'A':
                    self.state = 'B'
                else:
                    self.state = 'C'
            if self.state == 'A':
                self.last_update_loss = avg_loss
            self.total_loss = 0
            self.count = 0
        # state action
        if self.state == 'A':
            return tape.gradient(loss, hiddens), self.state
        else:
            return [], self.state


@register_strategy(
    strategy_name='split_state_async', backend='tensorflow', check_skip_grad=True
)
@proxy(PYUObject)
class PYUSLStateAsyncTFModel(SLStateAsyncTFModel):
    pass
