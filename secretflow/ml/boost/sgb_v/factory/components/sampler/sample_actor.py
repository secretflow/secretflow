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

from typing import List, Union, Tuple

import numpy as np
import math


# handle order map building for one party


class SampleActor:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def generate_one_partition_col_choices(
        self, colsample, feature_buckets: List[int]
    ) -> Tuple[Union[None, np.ndarray], int]:
        if colsample < 1:
            feature_num = len(feature_buckets)
            choices = math.ceil(feature_num * colsample)
            col_choices = np.sort(self.rng.choice(feature_num, choices, replace=False))

            buckets_count = 0
            for f_idx, f_buckets_size in enumerate(feature_buckets):
                if f_idx in col_choices:
                    buckets_count += f_buckets_size

            return col_choices, buckets_count
        else:
            return None, sum(feature_buckets)

    def goss(
        self, row_num: int, g: np.ndarray, top_rate: float, bottom_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        fact = (1 - top_rate) / bottom_rate
        top_N = math.ceil(top_rate * row_num)
        rand_N = math.ceil(bottom_rate * row_num)
        sorted_indices = np.argsort(abs(g.reshape(-1)))
        top_set = sorted_indices[:top_N]
        rand_set = self.rng.choice(sorted_indices[top_N:], rand_N, replace=False)
        # subsampling rows is public
        used_set = np.concatenate([top_set, rand_set])

        w = np.ones(row_num)
        w[rand_set] *= fact

        # shuffle to protect top set
        self.rng.shuffle(used_set)
        w = w[used_set]

        return used_set, w

    def generate_row_choices(
        self, row_num: int, rowsample_by_tree: float
    ) -> Union[None, np.ndarray]:
        if rowsample_by_tree == 1:
            return None
        sample_num_in_tree = math.ceil(row_num * rowsample_by_tree)

        sample_num, choices = (
            row_num,
            sample_num_in_tree,
        )

        return self.rng.choice(sample_num, choices, replace=False, shuffle=True)
