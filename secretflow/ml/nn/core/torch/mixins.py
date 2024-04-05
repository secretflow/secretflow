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

from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import StateDict


class ParametersMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._optimizers: List[optim.Optimizer] = []

    def optimizers(self) -> Union[optim.Optimizer, List[optim.Optimizer]]:
        if isinstance(self._optimizers, List) and len(self._optimizers) == 1:
            return self._optimizers[0]
        return self._optimizers

    def optimizers_state_dict(self) -> Union[List[StateDict], StateDict]:
        optimizers = self.optimizers()
        if isinstance(optimizers, optim.Optimizer):
            return optimizers.state_dict()
        else:
            return [opt.state_dict() for opt in optimizers]

    def load_optimizers_state_dict(self, stat_dict: Union[List[StateDict], StateDict]):
        optimizers = self.optimizers()
        if isinstance(optimizers, optim.Optimizer):
            optimizers.load_state_dict(stat_dict)
        else:
            assert isinstance(stat_dict, List) and len(stat_dict) == len(optimizers)
            for opt, sd in zip(optimizers, stat_dict):
                opt.load_state_dict(sd)

    def get_weights_not_bn(self, return_numpy=False):
        clean_state_dict = {}
        for k, v in self.state_dict().items():
            layername = k.split('.')[0]
            if not isinstance(getattr(self, layername), _BatchNorm):
                clean_state_dict[k] = torch.Tensor(v)
        if not return_numpy:
            return {k: v.cpu() for k, v in clean_state_dict.items()}
        else:
            weights_list = []
            for k, v in clean_state_dict.items():
                weights_list.append(v.cpu().numpy())
            return [e.copy() for e in weights_list]

    def update_weights_not_bn(self, weights):
        keys = self.state_dict().keys()
        weights_dict = {}
        for k, v in zip(keys, weights):
            layername = k.split('.')[0]
            if not isinstance(getattr(self, layername), _BatchNorm):
                weights_dict[k] = torch.Tensor(np.copy(v))

        self.load_state_dict(weights_dict, strict=False)

    def get_weights(self, return_numpy=False):
        if not return_numpy:
            return {k: v.cpu() for k, v in self.state_dict().items()}
        else:
            weights_list = []
            for v in self.state_dict().values():
                weights_list.append(v.cpu().numpy())
            return [e.copy() for e in weights_list]

    def update_weights(self, weights):
        keys = self.state_dict().keys()
        weights_dict = {}
        for k, v in zip(keys, weights):
            weights_dict[k] = torch.Tensor(np.copy(v))
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
                tensor_g = torch.from_numpy(np.array(g.copy()))
                p.grad = tensor_g.to(p.device)
