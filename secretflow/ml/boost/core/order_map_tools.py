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


import math
from typing import List, Tuple

import numpy as np


def skew_dist_split_points_search(x: np.ndarray, bucket_num: int) -> List:
    sorted_x = np.sort(x, axis=0)
    sorted_x_len = len(sorted_x)
    remained_count = sorted_x_len
    assert remained_count > 0, 'can not split empty x'

    value_category = list()
    last_value = None

    split_points = list()
    idx = 0
    expected_idx = math.ceil(remained_count / bucket_num)
    fast_skip = False
    while idx < sorted_x_len:
        v = sorted_x[idx]
        value_diff = v != last_value
        if not fast_skip and value_diff:
            if len(value_category) <= bucket_num:
                value_category.append(v)
            else:
                fast_skip = True
            last_value = v

        if idx >= expected_idx and value_diff:
            split_points.append(v)
            if len(split_points) == bucket_num - 1:
                break
            remained_count = sorted_x_len - idx
            expected_bin_count = math.ceil(
                remained_count / (bucket_num - len(split_points))
            )
            expected_idx = idx + expected_bin_count
            last_value = v

        if not fast_skip or idx >= expected_idx:
            idx += 1
        else:
            idx = expected_idx

    if len(value_category) <= bucket_num:
        # full dataset category count <= buckets
        # use category as split point.
        split_points = value_category[1:]
    elif split_points[-1] != sorted_x[-1]:
        # add max sample value into split_points like xgboost.
        split_points.append(sorted_x[-1])
    return split_points


def qcut(x: np.ndarray, bucket_num: int) -> Tuple[np.ndarray, List[float]]:
    """Compute a qcut on 1-d vector x into bucket num
    Percentile cut if it's ok. Use skew_dist_split_points_search otherwise.

    Args:
        x (np.ndarray): input to be cut into bins
        bucket_num (int): number of bins to cut

    Returns:
        Tuple[np.ndarray, List[float]]: digitized array and split points
    """
    quantiles = [i / bucket_num for i in range(1, bucket_num + 1, 1)]
    split_points = np.unique(np.quantile(x, quantiles))
    # any two quantiles turn out to be equal, we will have to do long tail trackled split points
    if len(split_points) >= bucket_num:
        split_points = split_points.astype(float).tolist()
    else:
        split_points = skew_dist_split_points_search(x, bucket_num)
        split_points = list(map(float, split_points))
    bins = np.digitize(x, split_points)
    return bins, split_points
