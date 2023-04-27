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
from typing import List

import numpy as np

from ..component import Component, Devices, print_params
from .leaf_actor import LeafActor


@dataclass
class LeafManagerParams:
    """
    'reg_lambda': float. L2 regularization term on weights.
        default: 0.1
        range: [0, 10000]
    'learning_rate': float, step size shrinkage used in update to prevent overfitting.
        default: 0.3
        range: (0, 1]
    """

    reg_lambda: float = 0.1
    learning_rate: float = 0.3


class LeafManager(Component):
    def __init__(self) -> None:
        self.params = LeafManagerParams()

    def show_params(self):
        print_params(self.params)

    def set_params(self, params: dict):
        reg_lambda = float(params.get('reg_lambda', 0.1))
        assert (
            reg_lambda >= 0 and reg_lambda <= 10000
        ), f"reg_lambda should in [0, 10000], got {reg_lambda}"

        lr = float(params.get('learning_rate', 0.3))
        assert lr > 0 and lr <= 1, f"learning_rate should in (0, 1], got {lr}"

        self.params.reg_lambda = reg_lambda
        self.params.learning_rate = lr

    def get_params(self, params: dict):
        params['reg_lambda'] = self.params.reg_lambda
        params['learning_rate'] = self.params.learning_rate

    def set_devices(self, devices: Devices):
        self.leaf_actor = LeafActor(device=devices.label_holder)

    def clear_leaves(self):
        self.leaf_actor.clear_leaves()

    def extend_leaves(
        self, pruned_node_selects: List[np.ndarray], pruned_node_indices: List[int]
    ):
        self.leaf_actor.extend_leaves(pruned_node_selects, pruned_node_indices)

    def get_leaf_selects(self):
        return self.leaf_actor.get_leaf_selects()

    def get_leaf_indices(self):
        return self.leaf_actor.get_leaf_indices()

    def compute_leaf_weights(self, g, h):
        return self.leaf_actor.compute_leaf_weights(
            self.params.reg_lambda, self.params.learning_rate, g, h
        )
