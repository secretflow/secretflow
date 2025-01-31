# Copyright 2024 Ant Group Co., Ltd.
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


class FedDYN(BaseTorchModel):
    def __init__(self):
        self.alpha = 0.1  # FedDYN algorithm hyperparameters, can be selected from [0.1, 0.01, 0.001]
        self.h = self.model.zeros_like()

    def train_step(
        self, weights: np.ndarray, cur_steps: int, train_steps: int, **kwargs
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
        if weights is None:
            assert (
                self.model is not None
            ), "Model cannot be none, please give model define"
            self.model.train()
            refresh_data = kwargs.get("refresh_data", False)
            if refresh_data:
                self._reset_data_iter()
            if weights is not None:
                self.model.update_weights(weights)
            num_sample = 0
            dp_strategy = kwargs.get("dp_strategy", None)
            logs = {}

            for _ in range(train_steps):
                self.optimizer.zero_grad()

                x, y, s_w = self.next_batch()
                num_sample += x.shape[0]
                y_pred = self.model(x)

                # do back propagation
                loss = self.loss(y_pred, y.long())
                loss.backward()
                self.optimizer.step()
                for m in self.metrics:
                    m.update(y_pred.cpu(), y.cpu())
            loss_value = loss.item()
            logs["train-loss"] = loss_value

            self.logs = self.transform_metrics(logs)
            self.wrapped_metrics.extend(self.wrap_local_metrics())
            self.epoch_logs = copy.deepcopy(self.logs)

            model_weights = self.model.get_weights(return_numpy=True)

            # DP operation
            if dp_strategy is not None:
                if dp_strategy.model_gdp is not None:
                    model_weights = dp_strategy.model_gdp(model_weights)

            return model_weights, num_sample
        self.initialize(self)
        assert self.model is not None, "Model cannot be none, please give model define"
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        # global parameters
        self.model.update_weights(
            weights
        )  # The local model is initialized to the global model
        src_model = copy.deepcopy(self.model)  # Record global model

        for p in src_model.parameters():
            p.requires_grad = False

        self.model.train()
        num_sample = 0
        dp_strategy = kwargs.get("dp_strategy", None)
        logs = {}

        for _ in range(train_steps):
            self.optimizer.zero_grad()
            iter_data = next(self.train_iter)
            if len(iter_data) == 2:
                x, y = iter_data
                s_w = None
            elif len(iter_data) == 3:
                x, y, s_w = iter_data
            x = x.float()
            num_sample += x.shape[0]
            if len(y.shape) == 1:
                y_t = y
            else:
                if y.shape[-1] == 1:
                    y_t = torch.squeeze(y, -1).long()
                else:
                    y_t = y.argmax(dim=-1)
            if self.use_gpu:
                x = x.to(self.exe_device)
                y_t = y_t.to(self.exe_device)
                if s_w is not None:
                    s_w = s_w.to(self.exe_device)
            y_pred = self.model(x)

            # do back propagation
            loss = self.loss(y_pred, y_t.long())

            l1 = loss  # first sub-formula L_k(theta)
            l2 = 0  # second sub-formula
            l3 = 0  # The third sub-formula ||theta - theta_t-1||^2
            for pgl, pm, ps in zip(
                self.gradL, self.model.parameters(), src_model.parameters()
            ):
                # pgl represents client gradient, pm represents client model, ps represents server model
                pgl = torch.Tensor(pgl)
                l2 += torch.dot(pgl.view(-1), pm.view(-1))
                l3 += torch.sum(torch.pow(pm - ps, 2))
            loss = l1 - l2 + 0.5 * self.alpha * l3

            loss.backward()
            self.optimizer.step()
            for m in self.metrics:
                m.update(y_pred.cpu(), y_t.cpu())

        # update grad_L
        new_gradL = []
        for pgl, pm, ps in zip(
            self.gradL, self.model.parameters(), src_model.parameters()
        ):
            pgl = torch.Tensor(pgl)
            ori_shape = pgl.size()
            pgl_tmp = pgl.view(-1) - self.alpha * (pm.view(-1) - ps.view(-1))
            pgl_tmp = pgl_tmp.view(ori_shape)
            new_gradL.append(pgl_tmp.detach().clone())

        self.h = self.h - self.alpha * 1.0 / self.num_clients * (
            self.model.parameters() - src_model.parameters()
        )
        self.model = self.model.parameters() - 1 / self.alpha * self.h
        self.gradL = new_gradL

        loss_value = loss.item()
        logs["train-loss"] = loss_value

        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        model_weights = self.model.get_weights(return_numpy=True)

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
            self.model.update_weights(weights)


@register_strategy(strategy_name="fed_dyn", backend="torch")
class PYUFedDYN(FedDYN):
    pass
