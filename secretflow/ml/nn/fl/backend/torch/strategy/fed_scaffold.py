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
from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy
import copy


class FedScaffold(BaseTorchModel):
    def train_step(
        self,
        weights: np.ndarray,
        server_control: np.ndarray,
        cur_steps: int,
        train_steps: int,
        **kwargs,
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
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
        if weights is not None:
            self.model.update_weights(weights)
        num_sample = 0
        dp_strategy = kwargs.get('dp_strategy', None)
        logs = {}
        x_weight=copy.deepcopy(self.model.get_weights(return_numpy=True))
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
            loss = self.loss(y_pred, y_t)
            loss.backward()
            self.optimizer.step(server_control,self.control)
            # self.control=copy.deepcopy(self.optimizer.get_grad())
            for m in self.metrics:
                m.update(y_pred.cpu(), y_t.cpu())
        loss_value = loss.item()
        logs['train-loss'] = loss_value

        self.logs = self.transform_metrics(logs)
        self.epoch_logs = copy.deepcopy(self.logs)

        model_weights = self.model.get_weights(return_numpy=True)
        self.update_scaffold(x_weight,model_weights,cur_steps,server_control)

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                model_weights = dp_strategy.model_gdp(model_weights)

        return model_weights, num_sample, self.control , self.delta_y,  self.delta_control
    
    def initial_scaffold(self):
        #增加控制变量         
        self.control = []
        self.delta_control = []
        self.delta_y=[]
        for layer_weight in self.model.get_weights(return_numpy=True):
            self.control.append(np.zeros_like(layer_weight))
            self.delta_control.append(np.zeros_like(layer_weight))
            self.delta_y.append(np.zeros_like(layer_weight))
    
    def update_scaffold(self,x,model_weights, local_steps, server_control):
        x_control=copy.deepcopy(self.control)
        for i, x_layer_weights in enumerate(x):
            self.control[i] =self.control[i] - server_control[i]+(x_layer_weights-model_weights[i])/(local_steps * self.optimizer.lr)
            self.delta_y[i] = model_weights[i]-x_layer_weights
            self.delta_control[i] = self.control[i] - x_control[i]

@register_strategy(strategy_name='fed_scaffold', backend='torch')
class PYUFedScaffold(FedScaffold):
    pass
