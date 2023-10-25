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
from numba import njit
import numpy as np


# regroup the roduct sums
def regroup_bucket_sums(bucket_sums_list, i):
    return np.concatenate([bucket_sums[i] for bucket_sums in bucket_sums_list], axis=0)


# if your dataset is small, numba may reduce the performance, you can disable numba by setting environment variable, os.environ["NUMBA_DISABLE_JIT"] = 1
# numba will speed up the code when the dataset is large and num boost round is large (>5).


# this function is used by other njit function, therefore should be njit annotated.
@njit
def select_sum(order_map_col, bucket, arr):
    subset = order_map_col == bucket
    return arr[subset].sum(axis=0)


@njit
def select_sums(order_map, bucket_num, node_select, arr):
    node_select = node_select.reshape(-1) == 1
    order_map = order_map[node_select]
    arr = arr[node_select]
    select_sums = np.empty((order_map.shape[1] * bucket_num, 2))
    for f in range(order_map.shape[1]):
        for bucket in range(bucket_num):
            index = f * bucket_num + bucket
            select_sums[index] = select_sum(order_map[:, f], bucket, arr)
    return cumsum_GH(select_sums, bucket_num)


def batch_select_sum(arr, children_nodes_selects: List, order_map, bucket_num):
    children_nodes_selects = np.array(children_nodes_selects)
    return batch_select_sum_inner(arr, children_nodes_selects, order_map, bucket_num)


@njit
def batch_select_sum_inner(
    arr, children_nodes_selects: np.ndarray, order_map, bucket_num
):
    return [
        select_sums(order_map, bucket_num, children_nodes_selects[i], arr)
        for i in range(np.int64(children_nodes_selects.shape[0]))
    ]


@njit
def cumsum_GH(x: np.ndarray, bucket_num: int):
    y = np.empty(x.shape)
    for i in range(0, x.shape[0], bucket_num):
        for j in range(x.shape[1]):
            y[i : i + bucket_num, j] = np.cumsum(x[i : i + bucket_num, j])
    return y
