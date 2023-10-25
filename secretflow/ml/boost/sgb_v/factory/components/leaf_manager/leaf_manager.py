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

from secretflow.ml.boost.sgb_v.core.params import default_params
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

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

    reg_lambda: float = default_params.reg_lambda
    learning_rate: float = default_params.learning_rate


class LeafManager(Component):
    def __init__(self) -> None:
        self.params = LeafManagerParams()

    def show_params(self):
        print_params(self.params)

    def set_params(self, params: dict):
        reg_lambda = params.get('reg_lambda', default_params.reg_lambda)
        lr = params.get('learning_rate', default_params.learning_rate)
        self.params.reg_lambda = reg_lambda
        self.params.learning_rate = lr

    def get_params(self, params: dict):
        params['reg_lambda'] = self.params.reg_lambda
        params['learning_rate'] = self.params.learning_rate

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder

    def set_actors(self, actors: List[SGBActor]):
        for actor in actors:
            if actor.device == self.label_holder:
                self.leaf_actor = actor
                break
        self.leaf_actor.register_class('LeafActor', LeafActor)

    def del_actors(self):
        del self.leaf_actor

    def clear_leaves(self):
        self.leaf_actor.invoke_class_method('LeafActor', 'clear_leaves')

    def extend_leaves(
        self, pruned_node_selects: List[np.ndarray], pruned_node_indices: List[int]
    ):
        self.leaf_actor.invoke_class_method(
            'LeafActor', 'extend_leaves', pruned_node_selects, pruned_node_indices
        )

    def get_leaf_selects(self):
        return self.leaf_actor.invoke_class_method('LeafActor', 'get_leaf_selects')

    def get_leaf_indices(self):
        return self.leaf_actor.invoke_class_method('LeafActor', 'get_leaf_indices')

    def compute_leaf_weights(self, g, h):
        reg_lambda = self.params.reg_lambda
        lr = self.params.learning_rate
        return self.leaf_actor.invoke_class_method(
            'LeafActor',
            'compute_leaf_weights',
            reg_lambda,
            lr,
            g,
            h,
        )
