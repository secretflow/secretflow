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

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
from heu import numpy as hnp


class SplitTree:
    """Each party will hold one split tree.
    Note this tree contains no leaf weights, but contains leaf indices.
    """

    def __init__(self) -> None:
        self.split_features = []
        self.split_values = []
        self.split_indices = []
        # split gains are not serialized and deserialized for now
        self.split_gains = []
        self.leaf_indices = []

    def is_empty(self) -> bool:
        return len(self.split_features) == 0

    def insert_split_node(
        self, feature: int, value: float, index: int, gain: Union[float, int]
    ) -> None:
        assert isinstance(feature, int), f"feature {feature}"
        assert isinstance(value, float), f"value {value}"
        assert isinstance(index, int), f"feature {index}"
        assert isinstance(gain, float) or isinstance(gain, int), f"gain {gain}"

        self.split_features.append(feature)
        self.split_values.append(value)
        self.split_indices.append(index)
        self.split_gains.append(gain)

    def extend_leaf_indices(self, leaf_indices: List[int]) -> None:
        self.leaf_indices.extend(leaf_indices)

    def predict_leaf_select(self, x: np.ndarray) -> np.ndarray:
        """
        compute leaf nodes' sample selects known by this partition.

        Args:
            x: dataset from this partition.
            tree: tree model store by this partition.

        Return:
            leaf nodes' selects: List[np.array], length equals leaf number.
        """
        x = x if isinstance(x, np.ndarray) else np.array(x)

        return hnp.tree_predict_with_indices(
            x,
            self.split_features,
            self.split_values,
            self.split_indices,
            self.leaf_indices,
        )

    def to_dict(self) -> Dict:
        return {
            "split_features": self.split_features,
            "split_values": self.split_values,
            "split_indices": self.split_indices,
            "leaf_indices": self.leaf_indices,
        }

    def gain_statistics(self) -> Tuple[Dict, Dict]:
        if len(self.split_gains) != len(self.split_features):
            logging.warning(
                "split gains does not support serialization and deserialization for now. train a new tree instead."
            )
            return {}, {}
        gain_sums = {}
        gain_counts = {}
        for i, feature in enumerate(self.split_features):
            if feature == -1:
                continue
            if feature not in gain_sums:
                gain_sums[feature] = 0.0
                gain_counts[feature] = 0
            gain_sums[feature] += self.split_gains[i]
            gain_counts[feature] += 1

        return gain_sums, gain_counts


def from_dict(dict: Dict) -> SplitTree:
    s = SplitTree()
    s.split_features = dict["split_features"]
    s.split_values = dict["split_values"]
    s.split_indices = dict["split_indices"]
    s.leaf_indices = dict["leaf_indices"]
    return s


def is_left_node(node_index: int) -> bool:
    """judge if a node is left node or right node from index
    root is view as left.
    """
    return node_index % 2 == 0
