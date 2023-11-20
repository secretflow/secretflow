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


from typing import List, Tuple, Union

import numpy as np

from ....core.distributed_tree.split_tree import SplitTree

# handle order map building for one party


class SplitTreeActor:
    def __init__(self, worker_index: int) -> None:
        self.worker_index = worker_index
        self.reset()

    def reset(self):
        self.tree = SplitTree()
        self.buckets_count = []
        self.feature_bucket = []

    def set_buckets_count(self, buckets_count: List[int]) -> None:
        """
        save how many buckets in each partition's all features.
        """
        self.buckets_count = buckets_count

    def set_feature_bucket(self, feature_bucket: List[int]):
        self.feature_bucket = feature_bucket

    def set_col_choices(self, col_choices: np.ndarray):
        self.col_choices = col_choices

    def predict_leaf_selects(self, x: np.ndarray) -> np.ndarray:
        return self.tree.predict_leaf_select(x)

    def tree_finish(self, leaf_indices: List[int]) -> SplitTree:
        self.tree.extend_leaf_indices(leaf_indices)
        return self.tree

    def find_split_bucket(self, split_bucket: int):
        """
        check if this partition contains split bucket.
        """
        pre_end_pos = 0
        for worker_index in range(len(self.buckets_count)):
            current_end_pod = pre_end_pos + self.buckets_count[worker_index]
            if split_bucket < current_end_pod:
                if worker_index == self.worker_index:
                    # split bucket is inside this partition's feature
                    return split_bucket - pre_end_pos
                else:
                    # split bucket is from other partition.
                    return -1
            pre_end_pos += self.buckets_count[worker_index]
        assert False, "should not be here, _is_primary_split"

    def split_buckets_to_paritition(self, split_buckets: List[int]) -> List[int]:
        return [self.find_split_bucket(split_bucket) for split_bucket in split_buckets]

    def get_split_feature(self, split_bucket: int) -> Union[None, Tuple[int, int]]:
        """
        find split bucket is belong to which feature.
        return feature index and bucket index
        """
        if split_bucket == -1:
            return None
        pre_end_pos = 0
        for f_idx in range(len(self.feature_bucket)):
            if self.col_choices is not None and f_idx not in self.col_choices:
                continue
            current_end_pod = pre_end_pos + self.feature_bucket[f_idx]
            if split_bucket < current_end_pod:
                return f_idx, split_bucket - pre_end_pos
            pre_end_pos += self.feature_bucket[f_idx]
        assert False, "should not be here, _get_split_feature"

    def get_split_feature_list_wise(
        self, split_buckets: List[int]
    ) -> List[Union[None, Tuple[int, int]]]:
        return [self.get_split_feature(split_bucket) for split_bucket in split_buckets]

    def do_split_list_wise(
        self,
        split_features: List[Tuple[int, int]],
        split_points: List[float],
        left_child_selects: List[np.ndarray],
        gain_is_cost_effective: List[bool],
        node_indices: List[int],
    ):
        """
        record split info and generate next level's left children select.
        """
        lchild_selects = []
        for key, s in enumerate(split_points):
            # pruning
            if not gain_is_cost_effective[key]:
                continue

            if s is not None:
                self.tree.insert_split_node(
                    split_features[key][0],
                    s,
                    node_indices[key],
                )
                # lchild' select
                lchild_selects.append(left_child_selects[key])
            else:
                self.tree.insert_split_node(-1, float("inf"), node_indices[key])
                lchild_selects.append(np.array([], dtype=np.uint8))

        return lchild_selects
