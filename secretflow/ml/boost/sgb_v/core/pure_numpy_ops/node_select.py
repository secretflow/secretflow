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


def root_select(samples: int) -> List[np.ndarray]:
    root_select = np.ones((1, samples), dtype=np.int8)
    return (root_select,)


def get_child_select(
    nodes_s: List[np.ndarray],
    lchilds_ss: List[np.ndarray],
    gain_is_cost_effective: List[bool],
    split_node_indices: List[int],
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    compute the next level's sample select indices, and node indices

    Args:
        nodes_s: sample select indices of each node from current level's nodes.
        lchilds_ss: left children's sample selects idx for current level's nodes (after pruning).
        A non-empty single sample select is a np.ndarray with the shape n_samples * 1
        and with entries being 0 and 1s. 1 indicates the sample remains in node.
        gain_is_cost_effective: List[bool]. indicate whether node should be split.
        split_node_indices: List[int]. node indices at the current level.

    Returns:
        sample select indices for nodes in the next level.
        node indices for the next level
        sample select indices for pruned nodes
        node indices for th pruned nodes
    """
    lchilds_ss = list(zip(*lchilds_ss))
    lchilds_s = [np.concatenate(ss, axis=None) for ss in lchilds_ss]
    childs_s = []
    node_indices = []
    index = 0
    pruned_s = []
    pruned_node_indices = []
    for i, current in enumerate(nodes_s):
        if not gain_is_cost_effective[i]:
            pruned_s.append(nodes_s[i])
            pruned_node_indices.append(split_node_indices[i])
            continue
        lchild = lchilds_s[index]
        assert (
            current.size == lchild.size
        ), "current size is {}, lchild size is {}".format(current.size, lchild.size)
        # current node's select mark with left child's select
        ls = current * lchild
        # get right child's select by sub.
        rs = current - ls
        node_index = split_node_indices[i]
        childs_s.extend([ls.astype(np.uint8), rs.astype(np.uint8)])
        l_index = 2 * node_index + 1
        node_indices.extend([l_index, l_index + 1])
        index += 1
    return childs_s, node_indices, pruned_s, pruned_node_indices


# TODO(zoupeicheng.zpc): These functions are experimental.
# improve efficiency of packing and unpacks by parallelization or other encoding method
# currently it is slow


def packbits_node_selects(node_selects: List[np.ndarray]) -> List[np.ndarray]:
    return [np.packbits(node_select) for node_select in node_selects]


def unpackbits_node_selects(
    node_selects_bits: List[np.ndarray],
    shape: Tuple[int],
):
    shape = np.array(shape)
    size = np.prod(shape)
    return [
        np.unpackbits(node_select_bits, count=size).reshape(shape)
        for node_select_bits in node_selects_bits
    ]


def unpack_node_select_lists(
    node_selects_bits: List[List[np.ndarray]], shape: Tuple[int]
):
    shape = np.array(shape)
    size = np.prod(shape)
    return [
        [
            np.unpackbits(node_select_bits, count=size).reshape(shape)
            if node_select_bits.size > 0
            else np.array([])
            for node_select_bits in node_select_bits_l
        ]
        for node_select_bits_l in node_selects_bits
    ]
