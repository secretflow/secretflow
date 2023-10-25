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


import collections
import copy
from typing import Tuple

import numpy as np
import tensorflow as tf

from secretflow.ml.nn.fl.backend.tensorflow.fl_base import BaseTFModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


class FedAvgW(BaseTFModel):
    """
    FedAvgW: A naive implementation of FedAvg, where the clients upload their trained model
    weights to the server for averaging and update their local models via the aggregated weights
    from the server in each federated round.
    """

    def train_step(
        self,
        weights: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[Any, int, Any, Any, Any]:
        """Accept ps model params, then do local train

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
        acc_weightss = copy.deepcopy(self.model.get_weights())
        V_local = copy.deepcopy(acc_weightss)
        s_local = copy.deepcopy(acc_weightss)
        for l in range(len(acc_weightss)):
            if len(acc_weightss[l].shape) == 2:
                U, s, V = self.svd(acc_weightss[l])
                V_local[l] = V
                s_local[l] = s
            else:
                weightss = acc_weightss[l]
                weight = weightss.flatten()
                weight = weight.reshape(2, len(weight) // 2)
                U, s, V = self.svd(weight)
                V_local[l] = V
                s_local[l] = s
        num_sample = 0
        dp_strategy = kwargs.get('dp_strategy', None)
        self.callbacks.on_train_batch_begin(cur_steps)
        logs = {}
        for _ in range(train_steps):

            iter_data = next(self.train_set)
            if len(iter_data) == 2:
                x, y = iter_data
                s_w = None
            elif len(iter_data) == 3:
                x, y, s_w = iter_data
            if isinstance(x, collections.OrderedDict):
                x = tf.stack(list(x.values()), axis=1)
            num_sample += x.shape[0]

            with tf.GradientTape() as tape:
                # Step 1: forward pass
                y_pred = self.model(x, training=True)
                # Step 2: loss calculation, the loss function is configured in `compile()`.
                loss = self.model.compiled_loss(
                    y,
                    y_pred,
                    regularization_losses=self.model.losses,
                    sample_weight=s_w,
                )
            # Step 3: back propagation
            trainable_vars = self.model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Step4: update metrics
            self.model.compiled_metrics.update_state(y, y_pred)
        for m in self.model.metrics:
            logs[m.name] = m.result().numpy()
        self.callbacks.on_train_batch_end(cur_steps + train_steps, logs)
        self.logs = logs
        self.epoch_logs = copy.deepcopy(self.logs)
        model_weights = self.model.get_weights()
        for l in range(len(model_weights)):
            if len(model_weights[l].shape) == 2:
                U, s, V = self.svd(model_weights[l])
                model_weights[l] = U

            else:
                weightss = model_weights[l]
                weight = weightss.flatten()
                weight = weight.reshape(2, len(weight) // 2)
                U, s, V = self.svd(weight)
                model_weights[l] = U

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                model_weights = dp_strategy.model_gdp(model_weights)

        return model_weights, num_sample, V_local, s_local, acc_weightss

    def svd(self, model_weights):
        U, s, V = np.linalg.svd(model_weights, full_matrices=False)
        return U, s, V


@register_strategy(strategy_name='fed_avg_w', backend='tensorflow')
class PYUFedAvgW(FedAvgW):
    pass
