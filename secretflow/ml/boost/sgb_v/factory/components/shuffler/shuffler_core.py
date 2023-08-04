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

import jax.tree_util
import numpy as np

from secretflow.ml.boost.sgb_v.core.pure_numpy_ops.random import (
    create_permuation_with_last_number_fixed,
)


class Shuffler:
    def __init__(self):
        self.reindex_list_map = {}

    def create_shuffle_mask(self, key: int, bucket_list: List[int]) -> List[int]:
        """
        Create a shuffle of buckets for key i and for each bucket,
        The random mask is a list of list of int.
        The random mask is private to each worker.
        It should be used to shuffle encrypted gh sum,
        before sending to label holder.
        It should also be applied to restore the correct split bucket,
        after receiving from label holder.

        The key is the index of node in a fixed level. (or node index for leaf wise mode)
        We will create one mask each for each fewer number child node selects at a level.
        Note when calculating bucket sums, we only do it for either left or right child.
        The other can be calculated

        Args:
            key: int. Each node select corresponds to one key.
            bucket_list: List[int]. List of number of buckets.

        """
        shuffle_mask = [
            create_permuation_with_last_number_fixed(feature_bucket_num).astype(int)
            for feature_bucket_num in bucket_list
        ]

        offset = 0
        reindex_list = []
        for feature_mask in shuffle_mask:
            feature_mask += offset
            reindex_list.extend(feature_mask)
            offset += feature_mask.size
        reindex_list = jax.tree_util.tree_map(
            lambda x: int(x) if isinstance(x, np.int64) else x, reindex_list
        )
        self.reindex_list_map[key] = reindex_list

    def get_shuffling_indices(self, key: int) -> List[int]:
        return self.reindex_list_map[key]

    def reset_shuffle_mask(self):
        self.reindex_list_map = {}

    def reset_shuffle_mask_with_keys(self, keys):
        for key in keys:
            self.reindex_list_map.pop(key)

    def is_shuffled(self) -> bool:
        return len(self.reindex_list_map) > 0

    def undo_shuffle_mask(self, key: int, index: int) -> int:
        return self.reindex_list_map[key][index]
