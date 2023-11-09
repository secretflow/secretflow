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
from numba import njit, prange
import numpy as np


# regroup the roduct sums
def regroup_bucket_sums(bucket_sums_list, i):
    return np.concatenate([bucket_sums[i] for bucket_sums in bucket_sums_list], axis=0)


# if your dataset is small, numba may reduce the performance, you can disable numba by setting environment variable, os.environ["NUMBA_DISABLE_JIT"] = 1
# numba will speed up the code when the dataset is large and num boost round is large (>5).


def batch_select_sum(arr, children_nodes_selects: List, order_map, bucket_num):
    """select sum

    Args:
        arr: array of shape (n, 2). n is the sample number.
        children_nodes_selects (List): List of length node number. Each element is a boolean array of size sample number,
            indicating whether the node is selected or not.
        order_map (np.ndarray): an array of shape (sample_number, feature_number), indicating which feature each sample belongs to.
        bucket_num (int): number of buckets in each feature

    Returns:
        bucket sums (List): return a list of length node number. Each element is an array of shape (order_map.shape[1] * bucket_num, 2)
    """
    children_node_select_one_arr_form = np.zeros(
        children_nodes_selects[0].shape, dtype=np.int64
    )
    node_num = len(children_nodes_selects)

    for i in range(node_num):
        children_node_select_one_arr_form += children_nodes_selects[i] * (i + 1)
    children_node_select_one_arr_form -= 1
    assert (
        np.max(children_node_select_one_arr_form) < node_num
    ), f"Make sure node selects have no intersection {children_node_select_one_arr_form}, {node_num}"
    bucket_sums_arr = batch_select_sum_inner(
        arr, children_node_select_one_arr_form, order_map, bucket_num, node_num
    )
    return [bucket_sums_arr[i].reshape(-1, 2) for i in range(node_num)]


@njit(parallel=True)
def batch_select_sum_inner(
    arr, children_node_select_one_arr_form, order_map, bucket_num, node_num
):
    feature_number = order_map.shape[1]
    sample_num = arr.shape[0]

    bucket_sums_arr = np.zeros(
        (node_num, feature_number, bucket_num, 2), dtype=arr.dtype
    )

    for j in prange(feature_number):
        for i in range(sample_num):
            node_index = children_node_select_one_arr_form[0, i]
            if node_index < 0:
                continue
            bucket_sums_arr[node_index, j, order_map[i, j]] += arr[i]

    for j in prange(feature_number):
        for n in range(node_num):
            for k in range(2):
                bucket_sums_arr[n, j, 0:bucket_num, k] = np.cumsum(
                    bucket_sums_arr[n, j, 0:bucket_num, k]
                )
    return bucket_sums_arr
