#!/usr/bin/env python3
# *_* coding: utf-8 *_*

# Copyright 2022 Ant Group Co., Ltd.
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


import math
from dataclasses import dataclass
from typing import List, Union

import numpy as np


@dataclass()
class Stats(object):
    """store information for each item in the summary

    Attributes:
        value: value of this stat
        w: weight of this stat
        delta: delta = rmax - rmin
    """

    value: float
    w: int
    delta: int


class QuantileSummaries(object):
    """QuantileSummary
        insert: insert data to summary
        merge: merge summaries
        fast_init: A fast version implementation creates the summary with little performance loss
        compress: compress summary to some size

    Attributes:
        compress_thres: if num of stats greater than compress_thres, do compress
        head_size: buffer size for insert data, when samples come to head_size do create summary
        error: 0 <= error < 1 default: 0.001, error tolerance for binning. floor((p - 2 * error) * N) <= rank(x) <= ceil((p + 2 * error) * N)
        abnormal_list: List of abnormal feature, will not participate in binning
    """

    def __init__(
        self,
        compress_thres: int = 10000,
        head_size: int = 10000,
        error: float = 1e-4,
        abnormal_list: List = None,
    ):
        self.compress_thres = compress_thres
        self.head_size = head_size
        self.error = error
        self.head_sampled = []
        self.sampled = []
        self.count = 0
        self.missing_count = 0
        if abnormal_list is None:
            self.abnormal_list = []
        else:
            self.abnormal_list = abnormal_list

    def fast_init(self, col_data: np.ndarray):
        if self.compress_thres > len(col_data):
            self.compress_thres = len(col_data)

        new_sampled = []
        for ab_item in self.abnormal_list:
            col_data = col_data[col_data != ab_item]
        bin_list = (
            np.linspace(0, len(col_data), self.compress_thres + 1)
            .round()[1:]
            .astype(int)
        )

        pre_rank = 0
        sorted_data = np.sort(col_data)
        for idx, bin_t in enumerate(bin_list):
            rank_t = sorted_data[bin_t - 1]
            delta = 0
            new_stats = Stats(rank_t, bin_t - pre_rank, delta)
            new_sampled.append(new_stats)
            pre_rank = bin_t
        self.sampled = new_sampled
        self.head_sampled = []
        self.count = len(col_data)
        if len(self.sampled) >= self.compress_thres:
            self.compress()

    def compress(self):
        """compress the summary, summary.sample will under compress_thres"""
        merge_threshold = 2 * self.error * self.count
        compressed = self._compress_immut(merge_threshold)
        self.sampled = compressed

    def query(self, quantile: float) -> float:
        """Use to query the value that specifies the quantile location

        Args:
            quantile : float [0.0, 1.0]
        Returns:
            float, the value of the quantile location
        """
        if self.head_sampled:
            self.compress()

        if quantile < 0 or quantile > 1:
            raise ValueError("Quantile should be in range [0.0, 1.0]")

        if self.count == 0:
            return 0

        if quantile <= self.error:
            return self.sampled[0].value

        if quantile >= 1 - self.error:
            return self.sampled[-1].value

        rank = math.ceil(quantile * self.count)
        target_error = math.ceil(self.error * self.count)
        min_rank = 0
        i = 1
        while i < len(self.sampled) - 1:
            cur_sample = self.sampled[i]
            min_rank += cur_sample.w
            max_rank = min_rank + cur_sample.delta
            if max_rank - target_error <= rank <= min_rank + target_error:
                return cur_sample.value
            i += 1
        return self.sampled[-1].value

    def value_to_rank(self, value: Union[float, int]) -> int:
        min_rank, max_rank = 0, 0
        for sample in self.sampled:
            if sample.value < value:
                min_rank += sample.w
                max_rank = min_rank + sample.delta
            else:
                return (min_rank + max_rank) // 2
        return (min_rank + max_rank) // 2

    def batch_query_value(self, values: List[float]) -> List[int]:
        """batch query function

        Args:
            values : List sorted_list of value. eg:[13, 56, 79]
        Returns:
            List : output ranks of each query
        """
        self.compress()
        res = []
        min_rank, max_rank = 0, 0
        idx = 0
        sample_idx = 0

        while sample_idx < len(self.sampled):
            v = values[idx]
            sample = self.sampled[sample_idx]
            if sample.value < v:
                min_rank += sample.w
                max_rank = min_rank + sample.delta
                sample_idx += 1
            else:
                res.append((min_rank + max_rank) // 2)
                idx += 1
                if idx >= len(values):
                    break

        while idx < len(values):
            res.append((min_rank + max_rank) // 2)
            idx += 1
        return res

    def _compress_immut(self, merge_threshold: float) -> List:
        if not self.sampled:
            return self.sampled

        res = []
        # Start from the last element
        head = self.sampled[-1]

        for i in range(len(self.sampled) - 2, 0, -1):
            this_sample = self.sampled[i]
            if this_sample.w + head.w + head.delta < merge_threshold:
                head.w = head.w + this_sample.w
            else:
                res.append(head)
                head = this_sample

        res.append(head)

        current_head = self.sampled[0]
        if current_head.value <= head.value and len(self.sampled) > 1:
            res.append(current_head)

        res.reverse()
        return res
