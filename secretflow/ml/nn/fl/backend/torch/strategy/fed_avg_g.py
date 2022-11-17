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
from secretflow.device import PYUObject, proxy
from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


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
        dp_strategy = kwargs.get('dp_strategy', None)

        if gradients is not None:
            # if gradients is not None, apply back propagation
            parameters = self.model.parameters()
            self.model.set_gradients(gradients, parameters)
            self.optimizer.step()

        num_sample = 0
        logs = {}
        local_gradients_sum = None

        for _ in range(train_steps):
            self.optimizer.zero_grad()
            iter_data = next(self.train_iter)
            if len(iter_data) == 2:
                x, y = iter_data
                s_w = None
            elif len(iter_data) == 3:
                x, y, s_w = iter_data

            num_sample += x.shape[0]
            y_t = y.argmax(dim=-1)
            y_pred = self.model(x)

            # do back propagation
            loss = self.loss(y_pred, y)
            loss.backward()
            local_gradients = self.model.get_gradients()

            if local_gradients_sum is None:
                local_gradients_sum = local_gradients
            else:
                local_gradients_sum += local_gradients

            for m in self.metrics:
                m.update(y_pred, y_t)
        loss = loss.item()
        logs['train-loss'] = loss
        self.logs = self.transform_metrics(logs)
        self.epoch_logs = copy.deepcopy(self.logs)

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                local_gradients_sum = dp_strategy.model_gdp(local_gradients_sum)

        return local_gradients_sum, num_sample


@register_strategy(strategy_name='fed_avg_g', backend='torch')
@proxy(PYUObject)
class PYUFedAvgG(FedAvgG):
    pass
