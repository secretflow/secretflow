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

from itertools import accumulate

import numpy as np
from heu import numpy as hnp

from secretflow.device import PYU, PYUObject
from secretflow.ml.boost.sgb_v.core.distributed_tree.distributed_tree import (
    DistributedTree,
)

from .core.pure_numpy_ops.pred import predict_tree_weight

# supports merging a distributed tree into a single tree, and thereby supports standalone deployment.
# WARNING: DO NOT USE THIS, UNLESS YOU KNOW EXACTLY WHAT THIS IS DOING.
# THIS FEATURE IS NOT SAFE. ALL INFO IN THE MODEL IS REVEALED TO OWNER PARTY.


class CompleteTree:
    """
    WARNING: DO NOT USE THIS, UNLESS YOU KNOW EXACTLY WHAT THIS IS DOING.
    THIS FEATURE IS NOT SAFE. ALL INFO IN THE MODEL IS REVEALED TO OWNER PARTY.
    """

    def __init__(self):
        self.split_features = []
        self.split_values = []
        self.split_indices = []
        self.leaf_indices = []
        self.leaf_weight = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predict on x

        Args:
            x: dataset from this partition.
            tree: tree model store by this partition.

        Return:
            leaf nodes' selects: List[np.array], length equals leaf number.

        """
        x = x if isinstance(x, np.ndarray) else np.array(x)

        leaf_selects = hnp.tree_predict_with_indices(
            x,
            self.split_features,
            self.split_values,
            self.split_indices,
            self.leaf_indices,
        )

        weight = self.leaf_weight
        return predict_tree_weight([leaf_selects], weight)

    def to_dict(self):
        return {
            "split_features": self.split_features,
            "split_values": self.split_values,
            "split_indices": self.split_indices,
            "leaf_indices": self.leaf_indices,
            "leaf_weight": self.leaf_weight.tolist(),
        }


def from_split_trees(split_trees, leaf_weight, partition_column_counts) -> CompleteTree:
    # TODO: add partition shapes, because feature indices are relative
    complete_tree = CompleteTree()
    node_number = len(split_trees[0].split_features)
    complete_tree.leaf_weight = leaf_weight

    complete_tree.leaf_indices.extend(split_trees[0].leaf_indices)
    partition_column_count_acc = list(
        accumulate(list(partition_column_counts.values()))
    )[:-1]

    partition_column_count_acc = [0] + partition_column_count_acc
    for i in range(node_number):
        for j, sp in enumerate(split_trees):
            if sp.split_features[i] != -1:
                complete_tree.split_features.append(
                    sp.split_features[i] + partition_column_count_acc[j]
                )
                complete_tree.split_values.append(sp.split_values[i])
                complete_tree.split_indices.append(sp.split_indices[i])
    return complete_tree


def from_dict(dict) -> CompleteTree:
    complete_tree = CompleteTree()
    complete_tree.split_features = dict["split_features"]
    complete_tree.split_values = dict["split_values"]
    complete_tree.split_indices = dict["split_indices"]
    complete_tree.leaf_indices = dict["leaf_indices"]
    complete_tree.leaf_weight = np.array(dict["leaf_weight"])
    return complete_tree


def from_distributed_tree(
    receive_party_pyu: PYU, distributed_tree: DistributedTree
) -> PYUObject:
    """Give all the tree model to a party_pyu and build a complete tree as a PYUObject"""
    split_trees = [
        sp.to(receive_party_pyu) for sp in distributed_tree.split_tree_dict.values()
    ]
    weights = distributed_tree.leaf_weight
    partition_column_counts = distributed_tree.partition_column_counts
    return receive_party_pyu(from_split_trees)(
        split_trees, weights, partition_column_counts
    )
