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


""" Async split learning strategy

"""

import copy
from typing import Callable, List

import tensorflow as tf

from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.sl.backend.tensorflow.sl_base import SLBaseTFModel
from secretflow.ml.nn.sl.strategy_dispatcher import register_strategy
from secretflow.security.privacy import DPStrategy


class SLAsyncTFModel(SLBaseTFModel):
    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        builder_fuse: Callable[[], tf.keras.Model],
        dp_strategy: DPStrategy,
        base_local_steps: int = 1,
        fuse_local_steps: int = 1,
        bound_param: float = 0.0,
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
        self.base_local_steps = base_local_steps
        self.fuse_local_steps = fuse_local_steps
        self.bound_param = bound_param

    @tf.function
    def _base_forward_internal(self, data_x, use_dp: bool = True, training=True):
        h = self.model_base(data_x, training=training)

        # Embedding differential privacy
        if use_dp and self.embedding_dp is not None:
            if isinstance(h, List):
                h = [self.embedding_dp(hi) for hi in h]
            else:
                h = self.embedding_dp(h)
        return h

    def base_backward(self, gradient):
        """backward on fusenet

        Args:
            gradient: gradient of fusenet hidden layer
        """

        for local_step in range(self.base_local_steps):
            return_hiddens = []
            with self.tape:
                if local_step == 0 and self.h is not None:
                    h = self.h
                else:
                    h = self._base_forward_internal(
                        self.data_x,
                        use_dp=False,
                        training=True,  # backward will only in training procedure
                    )
                if len(gradient) == len(h):
                    for i in range(len(gradient)):
                        return_hiddens.append(self.fuse_op(h[i], gradient[i]))
                else:
                    gradient = gradient[0]
                    return_hiddens.append(self.fuse_op(h, gradient))
                # add model.losses into graph
                return_hiddens.append(self.model_base.losses)

            trainable_vars = self.model_base.trainable_variables
            gradients = self.tape.gradient(return_hiddens, trainable_vars)
            self._base_backward_internal(gradients, trainable_vars)

        # clear intermediate results
        self.tape = None
        self.h = None
        self.data_x = None
        self.kwargs = {}

    def _fuse_net_train(self, hiddens, losses=[]):
        self.hiddens = copy.deepcopy(hiddens)
        return self._fuse_net_async_internal(
            hiddens,
            losses,
            self.train_y,
            self.train_sample_weight,
            self.fuse_local_steps,
            self.bound_param,
        )

    @tf.function
    def _fuse_net_async_internal(
        self,
        hiddens,
        losses,
        train_y,
        train_sample_weight,
        fuse_local_steps,
        bound_param,
    ):
        accumulated_gradients = []
        for local_step in range(fuse_local_steps):
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
            hidden_layer_gradients = tape.gradient(loss, hiddens)
            del tape
            lr = self.model_fuse.optimizer.lr
            # Step4: update metrics
            self.model_fuse.compiled_metrics.update_state(
                train_y, y_pred, sample_weight=train_sample_weight
            )
            # step5: accumulate gradients of embeddings
            if len(accumulated_gradients) == 0:
                accumulated_gradients = hidden_layer_gradients
            else:
                accumulated_gradients = [
                    tf.math.add(acc_g, h_g)
                    for acc_g, h_g in zip(accumulated_gradients, hidden_layer_gradients)
                ]
            # step6: update embeddings
            if fuse_local_steps > 1 and local_step < fuse_local_steps - 1:
                if local_step > 0:
                    hidden_layer_gradients = [
                        grad + bound_param * (layer_var - h)
                        for grad, layer_var, h in zip(
                            hidden_layer_gradients, hiddens, self.hiddens
                        )
                    ]
                hiddens = [
                    tf.math.subtract(layer_var, tf.math.multiply(lr, h_grad))
                    for layer_var, h_grad in zip(hiddens, hidden_layer_gradients)
                ]

        return accumulated_gradients


@register_strategy(strategy_name='split_async', backend='tensorflow')
@proxy(PYUObject)
class PYUSLAsyncTFModel(SLAsyncTFModel):
    pass
