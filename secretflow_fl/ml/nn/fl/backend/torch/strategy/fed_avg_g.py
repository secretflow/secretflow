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


class FedAvgG(BaseTorchModel):
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
        """Accept ps model params, then do local train

        Args:
            gradients: global gradients from params server
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """
        assert self.model is not None, "Model cannot be none, please give model define"
        assert (
            self.model.automatic_optimization
        ), "automatic_optimization must be True in FedAvgG"
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        dp_strategy = kwargs.get("dp_strategy", None)

        optimizer = self.model.optimizers()
        assert isinstance(
            optimizer, torch.optim.Optimizer
        ), "Only one optimizer is allowed in automatic optimization"

        if gradients is not None:
            # if gradients is not None, apply back propagation
            parameters = self.model.parameters()
            self.model.set_gradients(gradients, parameters)
            optimizer.step()

        num_sample = 0
        logs = {}
        local_gradients_sum = None
        loss: torch.Tensor = None

        for step in range(train_steps):
            optimizer.zero_grad()

            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]
            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)

            # do back propagation
            self.model.backward(loss)
            local_gradients = self.model.get_gradients()

            if local_gradients_sum is None:
                local_gradients_sum = local_gradients
            else:
                local_gradients_sum += local_gradients
        loss = loss.item()
        logs["train-loss"] = loss
        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                local_gradients_sum = dp_strategy.model_gdp(local_gradients_sum)
        # print(local_gradients_sum)
        return local_gradients_sum, num_sample

    def apply_weights(self, gradients, **kwargs):
        """Accept ps model gradients, then apply to model

        Args:
            gradients: global gradients from params server
        """
        if gradients is not None:
            parameters = self.model.parameters()
            self.model.set_gradients(gradients, parameters)
            optimizer = self.model.optimizers()
            optimizer.step()


@register_strategy(strategy_name="fed_avg_g", backend="torch")
class PYUFedAvgG(FedAvgG):
    pass
