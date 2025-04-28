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
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from secretflow_fl.ml.nn.fl.backend.tensorflow.fl_base import BaseTFModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy


class FedProx(BaseTFModel):
    """
    FedfProx: An FL optimization strategy that addresses the challenge of heterogeneity on data
    (non-IID) and devices, which adds a proximal term to the local objective function of each
    client, for better convergence. In the feature, this strategy will allow every client to
    train locally with a different Gamma-inexactness, for higher training efficiency.
    """

    def w_norm(
        self,
        w1: List,
        w2: List,
    ):
        w1_minus_w2 = tf.nest.map_structure(lambda a, b: a - b, w1, w2)
        return tf.linalg.global_norm(w1_minus_w2)

    def train_step(
        self, weights: np.ndarray, cur_steps: int, train_steps: int, **kwargs
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
        assert self.model is not None, "Model cannot be none, please give model define"
        if weights is not None:
            self.model.set_weights(weights)
        num_sample = 0
        dp_strategy = kwargs.get("dp_strategy", None)
        mu = kwargs.get("mu", 0.0)
        logs = {}

        for _ in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += self.get_sample_num(x)

            with tf.GradientTape() as tape:
                # Step 1: forward pass
                y_pred = self.model(x, training=True)
                # Step 2: loss calculation, the loss function is configured in `compile()`.
                loss = self.model.compute_loss(x, y, y_pred, s_w)
                # assumption: the compiled loss is the estimated empirical loss on per single sample
                if weights is not None:
                    # weights could be None in the very first step
                    proximal = tf.square(
                        self.w_norm(weights, self.model.trainable_variables)
                    )  # proximal = ||w - w^t||^2
                    # loss: adds the proximal term to the obj function
                    loss += mu / 2 * proximal
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

        model_weights = self.get_weights()

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                model_weights = dp_strategy.model_gdp(self.get_weights())

        return model_weights, num_sample

    def apply_weights(self, weights, **kwargs):
        """Accept ps model params,then apply to local model

        Args:
            updates: global updates from params server
        """
        if weights is not None:
            self.model.set_weights(weights)


@register_strategy(strategy_name="fed_prox", backend="tensorflow")
class PYUFedProx(FedProx):
    pass
