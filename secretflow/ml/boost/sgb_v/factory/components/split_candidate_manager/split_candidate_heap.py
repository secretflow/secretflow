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

import heapq
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SplitCandidateInfo:
    """Each node may contain the following information:
    1. sample selects (public)
    2. max gain (label holder private)
    3. split bucket index (public)
    """

    # indicate which samples are in this node
    sample_selects: np.ndarray
    # max_gain > gamma, because pruned nodes are not candidates.
    max_gain: float
    # the index of bucket corresponding to max gain
    split_bucket: int


class SplitCandidate:
    def __init__(
        self,
        node_index: int,
        sample_selects: np.ndarray,
        max_gain: float,
        split_bucket: int,
    ) -> None:
        self.node_index = node_index
        self.info = SplitCandidateInfo(sample_selects, max_gain, split_bucket)

    # for priority heap
    def __lt__(self, other: "SplitCandidate") -> bool:
        return self.info.max_gain > other.info.max_gain

    def __eq__(self, other: "SplitCandidate") -> bool:
        return self.info.max_gain == other.info.max_gain

    def __str__(self):
        return str(self.node_index)


class SplitCandidateHeap:
    def __init__(self):
        self.heap = []

    def push(
        self,
        node_index: int,
        sample_selects: np.ndarray,
        max_gain: float,
        split_bucket: int,
    ):
        heapq.heappush(
            self.heap,
            SplitCandidate(node_index, sample_selects, max_gain, split_bucket),
        )

    def batch_push(
        self,
        node_indices: List[int],
        node_sample_selects: List[np.ndarray],
        split_buckets: np.ndarray,
        split_gains: np.ndarray,
        gain_is_cost_effective: List[bool],
    ):
        for i, effective in enumerate(gain_is_cost_effective):
            if effective:
                self.push(
                    node_indices[i],
                    node_sample_selects[i],
                    split_gains[i],
                    split_buckets[i],
                )

    def pop(self) -> SplitCandidate:
        return heapq.heappop(self.heap)

    def is_heap_empty(self) -> bool:
        return len(self.heap) == 0

    # check heap not empty before extract
    def extract_best_split_info(self) -> Tuple[int, np.ndarray, int]:
        best_candidate = self.pop()
        return (
            best_candidate.node_index,
            best_candidate.info.sample_selects,
            best_candidate.info.split_bucket,
        )

    def extract_all_nodes(self) -> Tuple[List[int], List[np.ndarray]]:
        ids = [candidate.node_index for candidate in self.heap]
        sample_selects = [candidate.info.sample_selects for candidate in self.heap]
        self.heap = []
        return ids, sample_selects

    def reset(self):
        self.heap = []
