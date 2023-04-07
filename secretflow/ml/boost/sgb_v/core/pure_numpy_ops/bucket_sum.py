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

import numpy as np


def build_bin_indices_list(node_selects, order_map, bucket_num):
    return [
        np.argwhere((order_map[:, f] == bucket) & (node_selects.reshape(-1) == 1))
        .reshape(
            -1,
        )
        .tolist()
        for f in range(order_map.shape[1])
        for bucket in range(bucket_num)
    ]


def batch_select_sum(arr, children_nodes_selects, order_map, bucket_num):
    return [
        cumsum_GH(
            np.vstack(
                [
                    arr[indices].sum(axis=0)
                    for indices in build_bin_indices_list(
                        node_select, order_map, bucket_num
                    )
                ]
            ),
            bucket_num,
        )
        for node_select in children_nodes_selects
    ]


# regroup the roduct sums
def regroup_bucket_sums(bucket_sums_list, i):
    return np.concatenate([bucket_sums[i] for bucket_sums in bucket_sums_list], axis=0)


def cumsum_GH(x, bucket_num):
    y = np.zeros(x.shape)
    for i in range(0, x.shape[0], bucket_num):
        y[i : i + bucket_num] = np.cumsum(x[i : i + bucket_num], 0)
    return y
