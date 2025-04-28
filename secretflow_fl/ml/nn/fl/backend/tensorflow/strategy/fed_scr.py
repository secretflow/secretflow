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

import copy
from typing import Callable, List, Tuple

import numpy as np
import tensorflow as tf

from secretflow_fl.ml.nn.fl.backend.tensorflow.fl_base import BaseTFModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy
from secretflow_fl.utils.compressor import SCRSparse, sparse_encode


class FedSCR(BaseTFModel):
    """
    FedSCR: A structure-wise aggregation method to identify and remove redundant updates,
    it aggregates parameter updates over a particular structure (e.g., filters and channels).
    If the sum of the absolute updates of a model structure is lower than a given threshold,
    FedSCR will treat the updates in this structure as less important and filter them out.
    """

    def __init__(
        self,
        builder_base: Callable[[], tf.keras.Model],
        random_seed=None,
        **kwargs,
    ):
        super().__init__(builder_base, random_seed=random_seed, **kwargs)
        self._res = []

    def train_step(
        self,
        updates: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params,then do local train

        Args:
            updates: global updates from params server
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """

        def _add(matrices_a: List, matrices_b: List):
            results = [np.add(a, b) for a, b in zip(matrices_a, matrices_b)]
            return results

        dp_strategy = kwargs.get("dp_strategy", None)
        sparsity = kwargs.get("sparsity", 0.0)
        compressor = SCRSparse(sparsity)
        if updates is not None:
            current_weight = self.get_weights()
            server_weight = _add(current_weight, updates)
            self.model.set_weights(server_weight)
        num_sample = 0
        logs = {}
        self.model_weights = self.get_weights()
        for _ in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += self.get_sample_num(x)

            with tf.GradientTape() as tape:
                # Step 1: forward pass
                y_pred = self.model(x, training=True)
                # Step 2: loss calculation, the loss function is configured in `compile()`.
                loss = self.model.compute_loss(x, y, y_pred, s_w)
            # Step 3: back propagation
            trainable_vars = self.model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Step4: update metrics
            self.model.compute_metrics(x, y, y_pred, s_w)
        for m in self.model.metrics:
            logs[m.name] = m.result().numpy()
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.logs = logs
        self.epoch_logs = copy.deepcopy(self.logs)
        if self._res:
            client_updates = [
                np.add(np.subtract(new_w, old_w), res_u)
                for new_w, old_w, res_u in zip(
                    self.get_weights(), self.model_weights, self._res
                )
            ]
        else:
            # initial training res is zero
            client_updates = [
                np.subtract(new_w, old_w)
                for new_w, old_w in zip(self.get_weights(), self.model_weights)
            ]

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                client_updates_tensor = dp_strategy.model_gdp(client_updates)
                client_updates = [
                    client_updates_tensor[i].numpy() for i in range(len(client_updates))
                ]

        sparse_client_updates = compressor(client_updates)
        # compute residual updates
        self._res = [
            np.subtract(dense_u, sparse_u)
            for dense_u, sparse_u in zip(client_updates, sparse_client_updates)
        ]
        # do sparse encoding
        sparse_client_updates = sparse_encode(
            data=sparse_client_updates, encode_method="coo"
        )
        return sparse_client_updates, num_sample

    def apply_weights(self, updates, **kwargs):
        """Accept ps model params,then apply to local model

        Args:
            updates: global updates from params server
        """

        def _add(matrices_a: List, matrices_b: List):
            results = [np.add(a, b) for a, b in zip(matrices_a, matrices_b)]
            return results

        if updates is not None:
            current_weight = self.get_weights()
            server_weight = _add(current_weight, updates)
            self.model.set_weights(server_weight)


@register_strategy(strategy_name="fed_scr", backend="tensorflow")
class PYUFedSCR(FedSCR):
    pass
