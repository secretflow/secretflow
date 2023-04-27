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


from typing import List, Tuple

import numpy as np

from secretflow.device import PYUObject, proxy

from ..distributed_tree.split_tree import SplitTree
from .shuffler import Shuffler
from .splitter import Splitter


@proxy(PYUObject)
class SplitTreeTrainer:
    """
    It's main job is to produce the split tree hold by this party,
    and handle all book keepings in training.

    use in SGB model.
    do some compute works that only use one partition' dataset.
    """

    def __init__(self, idx: int) -> None:
        self.splitter = Splitter(idx)
        self.shuffler = Shuffler()

    def global_setup(self, x: np.ndarray, buckets: int, seed: int) -> np.ndarray:
        """
        Set up global context.
        """
        np.random.seed(seed)
        x = x if isinstance(x, np.ndarray) else np.array(x)
        self.splitter.build_maps(x, buckets)
        return self.splitter.get_order_map()

    def set_buckets_count(self, buckets_count: List[int]) -> None:
        """
        save how many buckets in each partition's all features.
        """
        self.splitter.set_buckets_count(buckets_count)

    def tree_setup(self, colsample: float) -> Tuple[np.ndarray, int]:
        """
        Set up tree context and do col sample if colsample < 1
        """
        self.tree = SplitTree()
        return self.splitter.set_up_col_choices(colsample)

    def predict_leaf_selects(self, x: np.ndarray) -> np.ndarray:
        return self.tree.predict_leaf_select(x)

    def tree_finish(self, leaf_indices: List[int]) -> SplitTree:
        self.tree.extend_leaf_indices(leaf_indices)
        return self.tree

    def do_split(
        self,
        split_buckets: List[int],
        sampled_rows: List[int],
        gain_is_cost_effective: List[bool],
        node_indices: List[int],
    ) -> List[np.ndarray]:
        """
        record split info and generate next level's left children select.
        """
        lchild_selects = []
        for key, s in enumerate(split_buckets):
            # pruning
            if not gain_is_cost_effective[key]:
                continue
            s = self.splitter.find_split_bucket(s)
            if s != -1:
                # unmask
                if self.shuffler.is_shuffled():
                    s = self.shuffler.undo_shuffle_mask(key, s)
                feature, split_point_idx = self.splitter.get_split_feature(s)
                self.tree.insert_split_node(
                    feature,
                    self.splitter.get_split_points()[feature][split_point_idx],
                    node_indices[key],
                )
                # lchild' select
                ls = self.splitter.compute_left_child_selects(
                    feature, split_point_idx, sampled_rows
                )
                lchild_selects.append(ls)
            else:
                self.tree.insert_split_node(-1, float("inf"), node_indices[key])
                lchild_selects.append(np.array([], dtype=np.int8))

        return lchild_selects

    def create_shuffle_mask(self, key: int) -> List[int]:
        col_choices = self.splitter.get_col_choices()
        if col_choices is not None:
            bucket_list = [
                self.splitter.get_feature_bucket_at(col_index)
                for col_index in col_choices
            ]
        else:
            bucket_list = self.splitter.get_feature_buckets()
        self.shuffler.create_shuffle_mask(key, bucket_list)
        return self.shuffler.get_shuffling_indices(key)

    def reset_shuffle_mask(self):
        self.shuffler.reset_shuffle_mask()
