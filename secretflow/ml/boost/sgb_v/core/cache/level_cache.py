# Copyright 2023 Ant Group Co., Ltd.
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
from typing import List
import numpy as np


# level cache is not embeded in split tree trainer because HEUObject in PYUObject is not well-defined.
class LevelCache:
    def __init__(self):
        self.level_nodes_GH = []
        self.cache = None

    def reset(self):
        self.cache = None
        self.level_nodes_GH = []

    def reset_level_nodes_GH(self):
        self.level_nodes_GH = []

    def collect_level_node_GH(self, child_GHL, idx, is_left):
        """collect one level node GH
        Args:
            child_GHL (PYUObject or HEUObject): PYUObject if self.pyu is not None.
            is_left (bool): whether this node is left child.
        """
        if is_left:
            self.level_nodes_GH.append(child_GHL)
            if self.cache is not None:
                cache = self.cache[idx] - child_GHL
                self.level_nodes_GH.append(cache)
        else:
            # right can only happens if not first level. i.e. cache must exist
            cache = self.cache[idx] - child_GHL
            self.level_nodes_GH.append(cache)
            self.level_nodes_GH.append(child_GHL)

    def collect_level_node_GH_level_wise(
        self, bucket_sums: List[np.ndarray], is_lefts: List[bool]
    ) -> None:
        for idx in range(len(is_lefts)):
            self.collect_level_node_GH(bucket_sums[idx], idx, is_lefts[idx])

    def update_level_cache(
        self, is_last_level: bool, gain_is_cost_effective: List[bool]
    ) -> None:
        # all nodes are pruned, this is effectively the last level.
        if sum(gain_is_cost_effective) == 0:
            self.cache = None
            return

        if not is_last_level:
            # cache pruning: only store cost effective caches.
            self.cache = [
                self.level_nodes_GH[i]
                for i in range(len(self.level_nodes_GH))
                if gain_is_cost_effective[i]
            ]
        elif self.cache:
            self.cache = None

    def get_level_nodes_GH(self) -> List[np.ndarray]:
        return self.level_nodes_GH
