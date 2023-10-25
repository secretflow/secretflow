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

from secretflow.device import PYUObject

from ....core.pure_numpy_ops.node_select import get_child_select, root_select
from ..component import Component, Devices


class NodeSelector(Component):
    def __init__(self) -> None:
        return

    def show_params(self):
        return

    def set_params(self, _: dict):
        return

    def get_params(self, _: dict):
        return

    def set_devices(self, devices: Devices):
        self.label_holder = devices.label_holder

    def set_actors(self, _):
        return

    def del_actors(self):
        return

    def root_select(self, sample_num):
        return root_select(samples=sample_num)

    def is_list_empty(self, any_list: Union[PYUObject, List]) -> PYUObject:
        return self.label_holder(lambda any_list: len(any_list) == 0)(any_list)

    def get_child_indices(self, node_indices: List[int], is_lefts: List[bool]):
        """node indices from the same level, in a [l_child, r_child, ...] fasion"""

        def _inner(node_indices, is_lefts):
            result = []
            for i, is_left in enumerate(is_lefts):
                is_left_offset = int(1 - is_left)
                result.append(node_indices[2 * i + is_left_offset])
            return result

        return self.label_holder(_inner)(node_indices, is_lefts)

    def get_pruned_indices_and_selects(
        self,
        node_indices: List[int],
        node_selects: List[np.ndarray],
        gain_is_cost_effective: List[bool],
    ) -> Tuple[PYUObject, PYUObject]:
        def _inner(
            node_indices, node_selects, gain_is_cost_effective
        ) -> Tuple[List[int], List[np.ndarray]]:
            pruned_indices = []
            pruned_node_selects = []
            for node_index, node_select, gain in zip(
                node_indices, node_selects, gain_is_cost_effective
            ):
                if not gain:
                    pruned_indices.append(node_index)
                    pruned_node_selects.append(node_select)
            return pruned_indices, pruned_node_selects

        return self.label_holder(_inner, num_returns=2)(
            node_indices, node_selects, gain_is_cost_effective
        )

    def pick_children_node_ss(
        self, node_select_list: PYUObject
    ) -> Tuple[List[PYUObject], List[bool], int]:
        return self.label_holder(pick_children_node_ss, num_returns=3)(node_select_list)

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
        ) = self.label_holder(get_child_select)(
            nodes_s, lchild_ss, gain_is_cost_effective, split_node_indices
        )

        return (
            split_node_selects,
            split_node_indices,
            pruned_node_selects,
            pruned_node_indices,
        )


def pick_children_node_ss(
    node_select_list: PYUObject,
) -> Tuple[List[PYUObject], List[bool], int]:
    """
    pick left/right children based on number of samples at each node.
    Args:
        node_select_list: PYUObject. List[np.ndarray] at label holder.
    Returns:
        children_node_select_list: List[PYUObject]. List[np.ndarray] at label holder.
        is_lefts: List[bool].
        node_num: int. len of node_select_list.
    """

    def choose_left_or_right_children(
        sums: List[np.ndarray], node_num: int
    ) -> List[bool]:
        if node_num == 1:
            is_lefts = [True]
        else:
            is_lefts = [sums[i] <= sums[i + 1] for i in range(node_num) if i % 2 == 0]
        return is_lefts

    def build_child_node_select_list(
        is_lefts: List[bool], nodes_s: List[np.ndarray], i: int
    ) -> np.ndarray:
        return nodes_s[i] if is_lefts[i // 2] else nodes_s[i + 1]

    sums = [np.sum(node_select) for node_select in node_select_list]

    node_num = len(node_select_list)
    is_lefts = choose_left_or_right_children(sums, node_num)
    children_nodes_s = []
    for i in range(node_num):
        if i % 2 == 0:
            children_nodes_s.append(
                build_child_node_select_list(is_lefts, node_select_list, i)
            )
    return children_nodes_s, is_lefts, node_num
