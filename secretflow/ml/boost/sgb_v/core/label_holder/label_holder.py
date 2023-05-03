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

from secretflow.device import PYUObject, proxy

from ..cache.level_cache import LevelCache
from ..preprocessing.params import LabelHolderInfo, RegType
from ..pure_numpy_ops.boost import compute_weight_from_node_select, find_best_splits
from ..pure_numpy_ops.bucket_sum import batch_select_sum, regroup_bucket_sums
from ..pure_numpy_ops.grad import compute_gh_linear, compute_gh_logistic, split_GH
from ..pure_numpy_ops.node_select import get_child_select, root_select
from ..pure_numpy_ops.pred import init_pred


@proxy(PYUObject)
class LabelHolder:
    """Label holder's computations in training.
    All responsibilities of label holder in training.
    """

    def __init__(self, label_holder_info: LabelHolderInfo) -> None:
        """
        Setup LabelHolder. Move parameters into train label holder.
        """
        self.reg_lambda = label_holder_info.reg_lambda
        self.gamma = label_holder_info.gamma
        self.learning_rate = label_holder_info.learning_rate
        self.sample_num = label_holder_info.sample_num
        self.subsample_rate = label_holder_info.subsample_rate
        self.obj_type = label_holder_info.obj_type
        self.base_score = label_holder_info.base_score

        self.rng = np.random.default_rng(label_holder_info.seed)

        self.sub_choices = None
        self.level_cache = LevelCache()
        self.leaf_node_selects = []
        self.leaf_node_indices = []

    def set_y(self, y: np.ndarray):
        self.y = y.reshape((y.shape[0], 1))

    def init_pred(self) -> np.ndarray:
        """Produce dummy prediction, used at the begin of training in SGB."""
        base = self.base_score
        sample_num = self.sample_num
        return init_pred(base=base, samples=sample_num)

    def root_select(self) -> List[np.ndarray]:
        """Produce initial node select at root, used at the begin of training in SGB."""
        sample_num = self.sample_num_in_tree
        return root_select(samples=sample_num)

    def get_child_select(
        self,
        nodes_s: List[np.ndarray],
        lchild_ss: List[np.ndarray],
        gain_is_cost_effective: List[bool],
        split_node_indices: List[int],
    ) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
        """
        compute the next level's sample select indices.

        Args:
            nodes_s: List[np.ndarray]. sample select indices of each node from current level's nodes.
            lchilds_ss: List[np.ndarray]. left children's sample selects idx for current level's nodes.
                A non-empty single sample select is a np.ndarray with the shape n_samples * 1
                and with entries being 0 and 1s. 1 indicates the sample remains in node.
            gain_is_cost_effective: List[bool]. indicate whether node should be split.
            split_node_indices: List[int]. node indices at the current level.

        Return:
            sample select indices for nodes in next tree level.
            node indices for the next level
            sample_selects for pruned nodes
            node_indices for pruned nodes
        """
        (
            split_node_selects,
            split_node_indices,
            pruned_node_selects,
            pruned_node_indices,
        ) = get_child_select(
            nodes_s, lchild_ss, gain_is_cost_effective, split_node_indices
        )

        return (
            split_node_selects,
            split_node_indices,
            pruned_node_selects,
            pruned_node_indices,
        )

    def extend_leaves(self, pruned_node_selects, pruned_node_indices):
        self.leaf_node_selects.extend(pruned_node_selects)
        self.leaf_node_indices.extend(pruned_node_indices)

    def clear_leaves(self):
        self.leaf_node_selects = []
        self.leaf_node_indices = []

    def get_leaf_indices(self) -> List[int]:
        return self.leaf_node_indices

    def is_list_empty(self, any_list: List) -> bool:
        return len(any_list) == 0

    def setup_context(self, pred: np.ndarray):
        """
        Set up pre-tree context.
        """

        self.sample_num_in_tree = math.ceil(self.sample_num * self.subsample_rate)
        assert (
            self.sample_num_in_tree > 0
        ), f"subsample {self.subsample_rate} is too small"
        if self.sample_num_in_tree < self.sample_num:
            rng, sample_num, choices = (
                self.rng,
                self.sample_num,
                self.sample_num_in_tree,
            )
            self.sub_choices = rng.choice(
                sample_num, choices, replace=False, shuffle=True
            )
            sub_choices = self.sub_choices
            pred = pred[sub_choices, :]
            y = self.y[sub_choices, :]
        else:
            y = self.y
        obj_type = self.obj_type
        gh = compute_gh(y, pred, obj_type)
        self.gh = np.concatenate([gh[0], gh[1]], axis=1)
        self.g = gh[0]
        self.h = gh[1]

    def get_gh(self):
        return self.gh

    def get_sub_choices(self):
        return self.sub_choices

    def pick_children_node_ss(
        self,
        node_select_list: PYUObject,
    ) -> Tuple[List[PYUObject], List[bool]]:
        """
        pick left/right children based on number of samples at each node.
        Args:
            node_select_list: PYUObject. List[np.ndarray] at label holder.
        Returns:
            children_node_select_list: List[PYUObject]. List[np.ndarray] at label holder.
            is_lefts: List[bool].
        """

        def choose_left_or_right_children(
            sums: List[np.ndarray], node_num: int
        ) -> List[bool]:
            if node_num == 1:
                is_lefts = [True]
            else:
                is_lefts = [
                    sums[i] <= sums[i + 1] for i in range(node_num) if i % 2 == 0
                ]
            return is_lefts

        def build_child_node_select_list(
            is_lefts: List[bool], nodes_s: List[np.ndarray], i: int
        ) -> np.ndarray:
            return nodes_s[i] if is_lefts[i // 2] else nodes_s[i + 1]

        sums = [np.sum(node_select) for node_select in node_select_list]

        self.node_num = len(node_select_list)
        node_num = self.node_num
        is_lefts = choose_left_or_right_children(sums, node_num)
        children_nodes_s = []
        for i in range(node_num):
            if i % 2 == 0:
                children_nodes_s.append(
                    build_child_node_select_list(is_lefts, node_select_list, i)
                )
        return children_nodes_s, is_lefts

    def do_leaf(self) -> np.ndarray:
        # see compute_weight_from_node_select.
        s = np.concatenate(self.leaf_node_selects, axis=0)
        reg_lambda = self.reg_lambda
        lr = self.learning_rate
        g = self.g
        h = self.h
        return compute_weight_from_node_select(s, g, h, reg_lambda, lr)

    def regroup_and_collect_level_nodes_GH(self, bucket_sums_list):
        node_num = self.node_num
        self.level_nodes_G, self.level_nodes_H = zip(
            *[
                split_GH(regroup_bucket_sums(bucket_sums_list, idx))
                for idx in range(node_num)
            ]
        )

    def reset_level_nodes_GH(self):
        self.level_cache.reset_level_nodes_GH()

    def collect_level_node_GH_level_wise(self, bucket_sums, is_lefts):
        self.level_cache.collect_level_node_GH_level_wise(bucket_sums, is_lefts)

    def update_level_cache(self, is_last_level, gain_is_cost_effective):
        self.level_cache.update_level_cache(is_last_level, gain_is_cost_effective)

    def get_level_nodes_GH(self):
        return self.level_cache.level_nodes_GH

    def batch_select_sum(
        self, arr, children_nodes_s, order_map, bucket_num
    ) -> PYUObject:
        return batch_select_sum(arr, children_nodes_s, order_map, bucket_num)

    def find_best_splits(self) -> Tuple[np.ndarray, np.ndarray]:
        G = self.level_nodes_G
        H = self.level_nodes_H
        reg_lambda = self.reg_lambda
        gamma = self.gamma
        split_buckets, should_split = find_best_splits(G, H, reg_lambda, gamma)
        return split_buckets, should_split


def compute_gh(
    y: np.ndarray, pred: np.ndarray, objective: RegType
) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute first and second order gradient of each sample.

    Args:
        y: sample true label of each sample.
        pred: prediction of each sample.
        objective: regression learning objective,

    Return:
        weight values.
    """
    if objective == RegType.Linear:
        return compute_gh_linear(y, pred)
    elif objective == RegType.Logistic:
        return compute_gh_logistic(y, pred)
    else:
        raise f"unknown objective {objective}"
