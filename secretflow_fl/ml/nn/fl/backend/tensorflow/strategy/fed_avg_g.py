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
from typing import Tuple

import numpy as np
import tensorflow as tf

from secretflow_fl.ml.nn.fl.backend.tensorflow.fl_base import BaseTFModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy


class FedAvgG(BaseTFModel):
    """
    FedAvgG: An implementation of FedAvg, where the clients upload their accumulated
    gradients during the federated round to the server for averaging and update their
    local models using the aggregated gradients from the server in each federated round.
    """

    def train_step(
        self,
        gradients: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params,then do local train

        Args:
            gradients: global gradients from params server
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters


        Returns:
            Parameters after local training
        """
        assert self.model is not None, "Model cannot be none, please give model define"
        dp_strategy = kwargs.get("dp_strategy", None)
        trainable_vars = self.model.trainable_variables
        if gradients is not None:
            # if gradients is not None, apply back propagation
            self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        num_sample = 0
        logs = {}

        local_gradients_sum = None
        for _ in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += self.get_sample_num(x)

            with tf.GradientTape() as tape:
                # Step 1: forward pass
                y_pred = self.model(x, training=True)
                # Step 2: loss calculation, the loss function is configured in `compile()`.
                loss = self.model.compute_loss(x, y, y_pred, s_w)
            # Step 3: compute local gradient
            local_gradients = tape.gradient(loss, trainable_vars)

            if local_gradients_sum is None:
                local_gradients_sum = local_gradients
            else:
                local_gradients_sum += local_gradients
            # Step4: update metrics
            self.model.compute_metrics(x, y, y_pred, s_w)

        for m in self.model.metrics:
            logs[m.name] = m.result().numpy()
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.logs = logs

        self.epoch_logs = copy.deepcopy(self.logs)

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                local_gradients_sum = dp_strategy.model_gdp(local_gradients_sum)
        return local_gradients_sum, num_sample

    def apply_weights(self, gradients, **kwargs):
        """Accept ps model params,then apply to local model

        Args:
            gradients: global gradients from params server
        """
        trainable_vars = self.model.trainable_variables
        if gradients is not None:
            # if gradients is not None, apply back propagation
            self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))


@register_strategy(strategy_name="fed_avg_g", backend="tensorflow")
class PYUFedAvgG(FedAvgG):
    pass
