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
from typing import Callable, Tuple

import numpy as np
import torch

from secretflow_fl.ml.nn.core.torch import BuilderType
from secretflow_fl.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow_fl.ml.nn.fl.strategy_dispatcher import register_strategy
from secretflow_fl.utils.compressor import STCSparse, sparse_decode, sparse_encode


class FedSTC(BaseTorchModel):
    """
    FedSTC: Sparse Ternary Compression (STC), a new compression framework that is speciﬁcally
    designed to meet the requirements of the Federated Learning environment. STC applies both
    sparsity and binarization in both upstream (client --> server) and downstream (server -->
    client) communication.
    """

    def __init__(self, builder_base: BuilderType, random_seed, skip_bn):
        super().__init__(builder_base, random_seed=random_seed, skip_bn=skip_bn)
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
                sparsity: SparsityParameters,the ratio of masked elements, default is 0.0
        Returns:
            Parameters after local training
        """
        assert self.model is not None, "Model cannot be none, please give model define"
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        dp_strategy = kwargs.get("dp_strategy", None)
        # prepare for the STC compression
        sparsity = kwargs.get("sparsity", 0.0)
        compressor = STCSparse(sparse_rate=sparsity)
        # update current weights
        if updates is not None:
            # Sparse matrix decoded in the downstream
            updates = sparse_decode(data=updates)
            weights = [np.add(w, u) for w, u in zip(self.model_weights, updates)]
            self.set_weights(weights)
        num_sample = 0
        logs = {}
        loss: torch.Tensor = None
        # store current weights for residual computing
        self.model_weights = self.get_weights()

        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]

            loss = self.model.training_step((x, y), cur_steps + step, sample_weight=s_w)

            if self.model.automatic_optimization:
                self.model.backward_step(loss)

        loss = loss.item()
        logs["train-loss"] = loss

        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        # do STC compression
        if self._res:
            client_updates = [
                np.add(np.subtract(new_w, old_w), res_u)
                for new_w, old_w, res_u in zip(
                    self.get_weights(),
                    self.model_weights,
                    self._res,
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
                    client_updates_tensor[i] for i in range(len(client_updates))
                ]
        # do sparsity + binarization
        sparse_client_updates = compressor(client_updates)
        # compute new residual
        self._res = [
            np.subtract(dense_u, sparse_u)
            for dense_u, sparse_u in zip(client_updates, sparse_client_updates)
        ]
        self.set_weights(self.model_weights)
        # do sparse encoding
        sparse_client_updates = sparse_encode(
            data=sparse_client_updates, encode_method="coo"
        )
        return sparse_client_updates, num_sample

    def apply_weights(self, updates, **kwargs):
        """Accept ps model params,then do local train

        Args:
            updates: global updates from params server
        """
        if updates is not None:
            # Sparse matrix decoded in the downstream
            updates = sparse_decode(data=updates)
            weights = [np.add(w, u) for w, u in zip(self.model_weights, updates)]
            self.set_weights(weights)


@register_strategy(strategy_name="fed_stc", backend="torch")
class PYUFedSTC(FedSTC):
    pass
