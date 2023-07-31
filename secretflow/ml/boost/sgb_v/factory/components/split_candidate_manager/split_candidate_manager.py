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

from typing import List, Tuple

import numpy as np
from secretflow.ml.boost.sgb_v.factory.sgb_actor import SGBActor

from .split_candidate_heap import SplitCandidateHeap
from ..component import Component, Devices


class SplitCandidateManager(Component):
    """Manages information and cache for split candidates.

    Split candidates are the nodes that are ready to be chosen for splits.

    In level wise booster, split candidates are current level nodes,
    and do not need complex management.

    In leaf wise booster, split candiates can be any node.
    Use split candidate manager to keep track of the node information.
    """

    def __init__(self) -> None:
        self.params = None
        self.label_holder = None
        self.heap = None

    def show_params(self):
        return

    def set_params(self, _: dict):
        return

    def get_params(self, _: dict):
        return

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder

    def set_actors(self, actors: List[SGBActor]):
        for actor in actors:
            if actor.device == self.label_holder:
                self.heap = actor
                break
        self.heap.register_class('SplitCandidateHeap', SplitCandidateHeap)

    def del_actors(self):
        del self.heap

    def batch_push(
        self,
        node_indices: List[int],
        node_sample_selects: List[np.ndarray],
        split_buckets: np.ndarray,
        split_gains: np.ndarray,
        gain_is_cost_effective: List[bool],
    ):
        self.heap.invoke_class_method(
            'SplitCandidateHeap',
            'batch_push',
            node_indices,
            node_sample_selects,
            split_buckets,
            split_gains,
            gain_is_cost_effective,
        )

    def push(
        self,
        node_index: int,
        sample_selects: np.ndarray,
        max_gain: float,
        split_bucket: int,
    ):
        self.heap.invoke_class_method(
            'SplitCandidateHeap',
            'push',
            node_index,
            sample_selects,
            max_gain,
            split_bucket,
        )

    def is_no_candidate_left(self) -> bool:
        return self.heap.invoke_class_method('SplitCandidateHeap', 'is_heap_empty')

    def extract_best_split_info(self) -> Tuple[int, np.ndarray, int]:
        return self.heap.invoke_class_method_three_ret(
            'SplitCandidateHeap', 'extract_best_split_info'
        )

    def extract_all_nodes(self) -> Tuple[List[int], List[np.ndarray]]:
        """Get all sample ids and sample selects and clean the heap"""
        return self.heap.invoke_class_method_two_ret(
            'SplitCandidateHeap', 'extract_all_nodes'
        )

    def reset(self):
        self.heap.invoke_class_method('SplitCandidateHeap', 'reset')
