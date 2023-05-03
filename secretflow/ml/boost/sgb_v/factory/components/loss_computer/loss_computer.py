# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from secretflow.device import PYUObject

from ....core.preprocessing.params import RegType
from ....core.pure_numpy_ops.grad import compute_gh_linear, compute_gh_logistic
from ..component import Component, Devices, print_params


@dataclass
class LossComputerParams:
    """
    'objective': Specify the learning objective.
        default: 'logistic'
        range: ['linear', 'logistic']
    """

    objective: RegType = RegType('logistic')


class LossComputer(Component):
    """Compute loss, gradients and hessians"""

    def __init__(self) -> None:
        self.params = LossComputerParams()

    def show_params(self):
        print_params(self.params)

    def set_params(self, params: dict):
        obj = params.get('objective', 'logistic')
        assert obj in [
            e.value for e in RegType
        ], f"objective should in {[e.value for e in RegType]}, got {obj}"
        obj = RegType(obj)
        self.params.objective = obj

    def get_params(self, params: dict):
        params['objective'] = self.params.objective

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder

    def compute_gh(
        self, y: Union[PYUObject, np.ndarray], pred: Union[PYUObject, np.ndarray]
    ) -> Tuple[PYUObject, PYUObject]:
        obj = self.params.objective
        if obj == RegType.Linear:
            g, h = self.label_holder(compute_gh_linear, num_returns=2)(y, pred)
            return g, h
        elif obj == RegType.Logistic:
            g, h = self.label_holder(compute_gh_logistic, num_returns=2)(y, pred)
            return g, h
        else:
            raise f"unknown objective {obj}"
