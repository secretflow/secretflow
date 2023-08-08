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

import pickle
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from secretflow.ml.boost.sgb_v.core.params import default_params

from ....core.pure_numpy_ops.boost import find_best_splits, find_best_splits_with_gains
from ..component import Component, Devices, print_params


@dataclass
class SplitFinderParams:
    """
    'gamma': float. Greater than 0 means pre-pruning enabled.
                    Gain less than it will not induce split node.
        default: 0.1
        range: [0, 10000]
    'reg_lambda': float. L2 regularization term on weights.
        default: 0.1
        range: [0, 10000]

    'audit_paths': dict. {device : path to save log for audit}
    """

    gamma: float = default_params.gamma
    reg_lambda: float = default_params.reg_lambda
    audit_paths: dict = field(default_factory=dict)


class SplitFinder(Component):
    def __init__(self) -> None:
        self.params = SplitFinderParams()

    def show_params(self):
        print_params(self.params)

    def set_params(self, params: dict):
        gamma = params.get('gamma', default_params.gamma)
        reg_lambda = params.get('reg_lambda', default_params.reg_lambda)
        audit_paths = params.get('audit_paths', {})

        self.params.gamma = gamma
        self.params.reg_lambda = reg_lambda
        self.params.audit_paths = audit_paths

    def get_params(self, params: dict):
        params['gamma'] = self.params.gamma
        params['reg_lambda'] = self.params.reg_lambda
        params['audit_path'] = self.params.audit_paths

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder

    def set_actors(self, _):
        return

    def del_actors(self):
        return

    def find_best_splits_with_gains(
        self, G: np.ndarray, H: np.ndarray, tree_num: int, leaf: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        reg_lambda = self.params.reg_lambda
        gamma = self.params.gamma
        split_buckets, split_gains, should_split = self.label_holder(
            find_best_splits_with_gains, num_returns=3
        )(G, H, reg_lambda, gamma)

        if self.label_holder.party in self.params.audit_paths:
            # may change log later
            split_info_path = (
                self.params.audit_paths[self.label_holder.party]
                + ".split_buckets.tree_"
                + str(tree_num)
                + ".leaf_"
                + str(leaf)
                + ".pickle"
            )
            # split info when considering the nth leaf
            self.label_holder(write_log)(split_buckets, split_info_path)
        return split_buckets, split_gains, should_split

    def find_best_splits_level_wise(
        self, G: np.ndarray, H: np.ndarray, tree_num: int, level: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        reg_lambda = self.params.reg_lambda
        gamma = self.params.gamma
        split_buckets, should_split = self.label_holder(
            find_best_splits, num_returns=2
        )(G, H, reg_lambda, gamma)

        if self.label_holder.party in self.params.audit_paths:
            split_info_path = (
                self.params.audit_paths[self.label_holder.party]
                + ".split_buckets.tree_"
                + str(tree_num)
                + ".level_"
                + str(level)
                + ".pickle"
            )

            self.label_holder(write_log)(split_buckets, split_info_path)
        return split_buckets, should_split


def write_log(x, path):
    with open(path, "wb") as f:
        pickle.dump(x, f)
