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
import numpy as np


def root_select(samples: int) -> List[np.ndarray]:
    root_select = np.ones((1, samples), dtype=np.int8)
    return (root_select,)


def get_child_select(
    nodes_s: List[np.ndarray], lchilds_ss: List[np.ndarray]
) -> List[np.ndarray]:
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
    lchilds_ss = list(zip(*lchilds_ss))
    lchilds_s = [np.concatenate(ss, axis=None) for ss in lchilds_ss]
    assert len(lchilds_s) == len(nodes_s), f"{len(lchilds_s)} != {len(nodes_s)}"
    childs_s = list()
    for current, lchild in zip(nodes_s, lchilds_s):
        assert (
            current.size == lchild.size
        ), "current size is {}, lchild size is {}".format(current.size, lchild.size)
        # current node's select mark with left child's select
        ls = current * lchild
        # get right child's select by sub.
        rs = current - ls
        childs_s.append(ls)
        childs_s.append(rs)
    return childs_s
