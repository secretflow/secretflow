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
import torch

from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy


class FedAvgW(BaseTorchModel):
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
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params, then do local train

        Args:
            weights: global weight from params server
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """
        assert self.model is not None, "Model cannot be none, please give model define"
        self.model.train()
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        if weights is not None:
            self.set_weights(weights)
        num_sample = 0
        dp_strategy = kwargs.get("dp_strategy", None)
        logs = {}
        loss: torch.Tensor = None

        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]

            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)

            if self.model.automatic_optimization:
                self.model.backward_step(loss)

        loss_value = loss.item()
        logs["train-loss"] = loss_value

        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        model_weights = self.get_weights(return_numpy=True)

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                model_weights = dp_strategy.model_gdp(model_weights)

        return model_weights, num_sample

    def apply_weights(self, weights, **kwargs):
        """Accept ps model params, then update local model

        Args:
            weights: global weight from params server
        """
        if weights is not None:
            self.set_weights(weights)


@register_strategy(strategy_name="fed_avg_w", backend="torch")
class PYUFedAvgW(FedAvgW):
    pass
