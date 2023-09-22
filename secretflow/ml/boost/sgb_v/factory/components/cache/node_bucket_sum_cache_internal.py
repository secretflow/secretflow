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


# node index: root has 0. if a node has index x, its left child is 2x+1. (right 2x+2).
class NodeCache:
    def __init__(self):
        self.cache = {}

    def reset(self):
        self.cache = {}

    def reset_node(self, node_index):
        if node_index in self.cache:
            self.cache.pop(node_index)

    def collect_node_bucket_sum(self, node_index: int, bucket_sum: int):
        """Collect one node's bucket sum.
        If its parent exist, calculate and cache the bucket sum for another child, and parent cache removed.
        """
        parent_index = (node_index - 1) // 2
        self.cache[node_index] = bucket_sum
        if parent_index >= 0 and parent_index in self.cache:
            parent_index_double = 2 * parent_index
            other_index = (
                node_index - parent_index_double
            ) * 2 % 3 + parent_index_double
            parent_bucket_sum = self.cache.pop(parent_index)
            self.cache[other_index] = parent_bucket_sum - bucket_sum

    def batch_collect_node_bucket_sums(
        self, node_indices: List[int], bucket_sums: List[int]
    ):
        for node_index, bucket_sum in zip(node_indices, bucket_sums):
            self.collect_node_bucket_sum(node_index, bucket_sum)

    def get_node(self, node_index):
        return self.cache.get(node_index)
