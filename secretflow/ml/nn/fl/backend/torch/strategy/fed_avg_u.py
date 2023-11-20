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

from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


class FedAvgU(BaseTorchModel):
    """
    FedAvgU: An implementation of FedAvg, where the clients upload their model updates
    to the server for averaging and update their local models with the aggregated
    updates from the server in each federated round. This paradigm acts the same as
    FedAvgG when using the SGD optimizer, but may not for other optimizers (e.g., Adam).
    """

    def train_step(
        self,
        updates: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params, then do local train

        Args:
            updates: global updates from params server
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        dp_strategy = kwargs.get('dp_strategy', None)
        if updates is not None:
            weights = [np.add(w, u) for w, u in zip(self.get_weights(), updates)]
            self.set_weights(weights)

        num_sample = 0
        logs = {}
        model_weights = self.get_weights()
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

            if self.use_gpu:
                x = x.to(self.exe_device)
                y_t = y_t.to(self.exe_device)
                if s_w is not None:
                    s_w = s_w.to(self.exe_device)

            y_pred = self.model(x)

            # do back propagation
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()
            for m in self.metrics:
                m.update(y_pred.cpu(), y_t.cpu())
        loss = loss.item()
        logs['train-loss'] = loss
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.logs = self.transform_metrics(logs)
        self.epoch_logs = copy.deepcopy(self.logs)

        client_updates = [
            np.subtract(new_w, old_w)
            for new_w, old_w in zip(self.get_weights(), model_weights)
        ]

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                client_updates = dp_strategy.model_gdp(client_updates)

        return client_updates, num_sample

    def apply_weights(self, updates, **kwargs):
        """Accept ps model params, then apply to local model

        Args:
            updates: global updates from params server
        """
        if updates is not None:
            weights = [np.add(w, u) for w, u in zip(self.get_weights(), updates)]
            self.set_weights(weights)


@register_strategy(strategy_name='fed_avg_u', backend='torch')
class PYUFedAvgU(FedAvgU):
    pass
