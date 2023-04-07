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
from typing import Any, Dict, List, Tuple

import numpy as np

from secretflow.device import PYUObject, reveal

from .level_cache import LevelCache
from .params import LabelHolderInfo, RegType
from .pure_numpy_ops.boost import (
    compute_obj,
    compute_weight_from_node_select,
    find_best_splits,
)
from .pure_numpy_ops.bucket_sum import batch_select_sum, regroup_bucket_sums
from .pure_numpy_ops.grad import compute_gh_linear, compute_gh_logistic, split_GH
from .pure_numpy_ops.node_select import get_child_select, root_select
from .pure_numpy_ops.pred import init_pred


class LabelHolder:
    """Label holder's computations in training.
    All responsibilities of label holder in training.
    """

    def __init__(self, label_holder_info: LabelHolderInfo) -> Dict[str, Any]:
        """
        Setup LabelHolder. Move parameters into train label holder.
        """
        self.label_holder = label_holder_info.label_holder_pyu
        self.heu = label_holder_info.heu
        self.reg_lambda = label_holder_info.reg_lambda
        self.learning_rate = label_holder_info.learning_rate
        self.sample_num = label_holder_info.sample_num
        self.subsample_rate = label_holder_info.subsample_rate
        self.obj_type = label_holder_info.obj_type
        self.base_score = label_holder_info.base_score

        self.y = self.label_holder(lambda y: y.reshape((1, y.shape[0])))(
            label_holder_info.y
        )
        self.rng = self.label_holder(np.random.default_rng)(label_holder_info.seed)

        self.sub_choices = None
        self.level_cache = LevelCache(self.label_holder)

    def init_pred(self) -> PYUObject:
        """Produce dummy prediction, used at the begin of training in SGB."""
        base = self.base_score
        sample_num = self.sample_num
        return self.label_holder(init_pred, static_argnames=('base', 'samples'))(
            base=base, samples=sample_num
        )

    def root_select(self) -> PYUObject:
        """Produce initial node select at root, used at the begin of training in SGB."""
        return self.label_holder(root_select, static_argnames=('samples',))(
            samples=self.sample_num_in_tree
        )

    def get_child_select(self, nodes_s, lchild_ss) -> PYUObject:
        """
        compute the next level's sample select indices.

        Args:
            nodes_s: sample select indices of each node from current level's nodes.
            lchilds_ss: left children's sample selects idx for current level's nodes.
            A non-empty single sample select is a np.ndarray with the shape n_samples * 1
            and with entries being 0 and 1s. 1 indicates the sample remains in node.

        Return:
            sample select indices for nodes in next tree level.
        """
        return self.label_holder(get_child_select)(nodes_s, lchild_ss)

    def compute_weight_from_node_select(self, node_select: np.ndarray) -> PYUObject:
        """
        compute weight values of tree leaf nodes.

        Args:
            node_select: sample selects in each leaf node.

        Return:
            weight values.
        """
        return self.label_holder(compute_weight_from_node_select)(
            node_select, self.g, self.h, self.reg_lambda, self.learning_rate
        )

    def setup_context(self, pred: PYUObject):
        """
        Set up pre-tree context.
        """

        self.sample_num_in_tree = math.ceil(self.sample_num * self.subsample_rate)
        assert (
            self.sample_num_in_tree > 0
        ), f"subsample {self.subsample_rate} is too small"
        if self.sample_num_in_tree < self.sample_num:
            self.sub_choices = self.label_holder(
                lambda rng, sample_num, choices: rng.choice(
                    sample_num, choices, replace=False, shuffle=True
                )
            )(self.rng, self.sample_num, self.sample_num_in_tree)
            pred = self.label_holder(lambda pred, sub_choices: pred[:, sub_choices])(
                pred, self.sub_choices
            )
            y = self.label_holder(lambda y, sub_choices: y[:, sub_choices])(
                self.y, self.sub_choices
            )
        else:
            y = self.y
        gh = self.label_holder(compute_gh)(y, pred, self.obj_type)
        t = self._concatenate([gh[0], gh[1]], axis=0)
        self.gh_t = self.label_holder(lambda m: m.transpose())(t)
        self.g = self._select(gh, 0)
        self.h = self._select(gh, 1)

    def pick_children_node_ss(
        self, node_select_list: PYUObject
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
            if node_num <= 1:
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

        sums = self.label_holder(
            lambda node_select_list: [
                np.sum(node_select) for node_select in node_select_list
            ]
        )(node_select_list)
        self.node_num = reveal(
            self.label_holder(lambda node_select_list: len(node_select_list))(
                node_select_list
            )
        )
        node_num = self.node_num
        is_lefts = reveal(
            self.label_holder(choose_left_or_right_children)(sums, node_num)
        )
        children_nodes_s = []
        for i in range(node_num):
            if i % 2 == 0:
                children_nodes_s.append(
                    self.label_holder(build_child_node_select_list)(
                        is_lefts, node_select_list, i
                    )
                )
        return children_nodes_s, is_lefts

    def do_leaf(self, ss: PYUObject) -> np.ndarray:
        """ss is PYUObject with type List[np.ndarray] at label holder."""
        # see compute_weight_from_node_select.
        s = self._concatenate(ss, axis=0)
        return self.compute_weight_from_node_select(s)

    def _select(self, x, item):
        return self.label_holder(lambda x, item: x[item])(x, item)

    def _concatenate(self, arr_like, axis: int = None):
        return self.label_holder(lambda l, axis: np.concatenate(l, axis=axis))(
            arr_like, axis
        )

    def _subtract(
        self,
        a,
        b,
    ):
        return self.label_holder(lambda a, b: a - b)(a, b)

    def _compute_obj(self, G, H):
        return self.label_holder(compute_obj)(G, H, self.reg_lambda)

    def _argmax(self, arr, axis):
        return self.label_holder(lambda arr, axis: np.argmax(arr, axis))(arr, axis)

    def regroup_and_collect_level_nodes_GH(self, bucket_sums_list):
        node_num = self.node_num
        self.level_nodes_G, self.level_nodes_H = zip(
            *[
                self.label_holder(
                    lambda bucket_sums_list, idx: split_GH(
                        regroup_bucket_sums(bucket_sums_list, idx)
                    ),
                    num_returns=2,
                )(bucket_sums_list, idx)
                for idx in range(node_num)
            ]
        )

    def reset_level_nodes_GH(self):
        self.level_cache.reset_level_nodes_GH()

    def collect_level_node_GH(self, child_GHL, idx, is_left):
        self.level_cache.collect_level_node_GH(
            self.label_holder(lambda x, y: x[y])(child_GHL, idx), idx, is_left
        )

    def update_level_cache(self, is_last_level):
        self.level_cache.update_level_cache(is_last_level)

    def get_level_nodes_GH(self):
        return self.level_cache.level_nodes_GH

    def batch_select_sum(
        self, arr, children_nodes_s, order_map, bucket_num
    ) -> PYUObject:
        return self.label_holder(batch_select_sum)(
            arr, children_nodes_s, order_map, bucket_num
        )

    def find_best_splits(self):
        return self.label_holder(find_best_splits)(
            self.level_nodes_G, self.level_nodes_H, self.reg_lambda
        )


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
