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


from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from torchmetrics import Metric

import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss as BaseTorchLoss


class BaseModule(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    def get_weights(self, return_numpy=False):
        if not return_numpy:
            return {k: v.cpu() for k, v in self.state_dict().items()}
        else:
            weights_list = []
            for v in self.state_dict().values():
                weights_list.append(v.cpu().numpy())
            return [e.copy() for e in weights_list]

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def update_weights(self, weights):
        keys = self.state_dict().keys()
        weights_dict = {}
        for k, v in zip(keys, weights):
            weights_dict[k] = torch.Tensor(v)
        self.load_state_dict(weights_dict)

    def get_gradients(self, parameters=None):
        if parameters is None:
            parameters = self.parameters()
        grads = []
        for p in parameters:
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return [g.copy() for g in grads]

    def set_gradients(
        self,
        gradients: List[Union[torch.Tensor, np.ndarray]],
        parameters: Optional[List[torch.Tensor]] = None,
    ):
        if parameters is None:
            parameters = self.parameters()
        for g, p in zip(gradients, parameters):
            if g is not None:
                p.grad = torch.from_numpy(np.array(g.copy()))


# @dataclass
class TorchModel:
    def __init__(
        self,
        model_fn: BaseModule = None,
        loss_fn: BaseTorchLoss = None,
        optim_fn: optim.Optimizer = None,
        metrics: List[Metric] = [],
    ):

        self.model_fn = model_fn
        self.loss_fn: BaseTorchLoss = loss_fn
        self.optim_fn: optim.Optimizer = optim_fn
        self.metrics: List[Metric] = metrics
