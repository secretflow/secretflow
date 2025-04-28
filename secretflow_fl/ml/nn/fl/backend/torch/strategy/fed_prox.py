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
import torch

from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy


class FedProx(BaseTorchModel):
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
        l1 = len(w1)
        assert l1 == len(w2), "weights should be same in the shape"
        proximal_term = 0
        for i in range(l1):
            proximal_term += (torch.tensor(w1[i]) - w2[i]).norm(2) ** 2
        return proximal_term

    def train_step(
        self,
        weights: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params,then do local train

        Args:
            weights: global weight from params server
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
                mu: hyper-parameter for the proximal term, default is 0.0
        Returns:
            Parameters after local training
        """
        assert self.model is not None, "Model cannot be none, please give model define"
        assert (
            self.model.automatic_optimization
        ), "automatic_optimization must be True in FedProx"
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        if weights is not None:
            self.set_weights(weights)
        num_sample = 0
        dp_strategy = kwargs.get("dp_strategy", None)
        logs = {}
        loss: torch.Tensor = None

        mu = kwargs.get("mu", 0.0)

        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]

            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)
            if weights is not None:
                w_norm = self.w_norm(weights, list(self.model.parameters()))
                loss += mu / 2 * w_norm

            self.model.backward_step(loss)

        loss = loss.item()
        logs["train-loss"] = loss

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
        """Accept ps model params,then update local model

        Args:
            weights: global weight from params server
        """
        if weights is not None:
            self.set_weights(weights)


@register_strategy(strategy_name="fed_prox", backend="torch")
class PYUFedProx(FedProx):
    pass
