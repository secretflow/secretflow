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


from typing import List

import numpy as np

from ....core.pure_numpy_ops.boost import compute_weight_from_node_select

# handle order map building for one party


class LeafActor:
    def __init__(self) -> None:
        self.leaf_node_selects = []
        self.leaf_node_indices = []

    def extend_leaves(
        self, pruned_node_selects: List[np.ndarray], pruned_node_indices: List[int]
    ):
        self.leaf_node_selects.extend(pruned_node_selects)
        self.leaf_node_indices.extend(pruned_node_indices)

    def clear_leaves(self):
        self.leaf_node_selects = []
        self.leaf_node_indices = []

    def get_leaf_indices(self) -> List[int]:
        return self.leaf_node_indices

    def get_leaf_selects(self) -> List[np.ndarray]:
        return self.leaf_node_selects

    def compute_leaf_weights(self, reg_lambda, lr, g, h):
        s = np.concatenate(self.leaf_node_selects, axis=0)
        return compute_weight_from_node_select(s, g, h, reg_lambda, lr)
