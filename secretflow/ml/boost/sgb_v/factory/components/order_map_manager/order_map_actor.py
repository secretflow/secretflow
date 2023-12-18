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

from .order_map_context import OrderMapContext


# handle order map building for one party
class OrderMapActor:
    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.ordermap_context = OrderMapContext()

    def build_order_map(self, x: np.ndarray, buckets: int, seed: int) -> np.ndarray:
        """
        Set up global context.
        """
        np.random.seed(seed)
        x = np.array(x, order='F')
        self.ordermap_context.build_maps(x, buckets)
        return self.ordermap_context.get_order_map()

    def get_features(self) -> int:
        """Get the number of features at this partition"""
        return self.ordermap_context.get_features()

    def get_feature_buckets(self) -> List[int]:
        """Get the number of buckets for each feature"""
        return self.ordermap_context.get_feature_buckets()

    def get_split_points(self) -> List[List[float]]:
        return self.ordermap_context.get_split_points()

    def batch_query_split_points(
        self, queries: List[Union[None, Tuple[int, int]]]
    ) -> Union[None, float]:
        split_points = self.get_split_points()
        return [split_points[q[0]][q[1]] if q is not None else None for q in queries]

    def get_bucket_list(self, col_choices: List[int]) -> List[int]:
        """Get number of buckets at chosen columns"""
        if col_choices is not None:
            bucket_list = [
                self.ordermap_context.get_feature_bucket_at(col_index)
                for col_index in col_choices
            ]
        else:
            bucket_list = self.get_feature_buckets()
        return bucket_list

    def batch_compute_left_child_selects(
        self,
        split_feature_buckets: List[Union[None, Tuple[int, int]]],
        sampled_indices: Union[List[int], None] = None,
    ) -> List[Union[None, np.ndarray]]:
        return [
            self.compute_left_child_selects(
                split_feature_bucket[0], split_feature_bucket[1], sampled_indices
            )
            if split_feature_bucket is not None
            else None
            for split_feature_bucket in split_feature_buckets
        ]

    def compute_left_child_selects(
        self,
        feature: int,
        split_point_index: int,
        sampled_indices: Union[List[int], None] = None,
    ) -> np.ndarray:
        """Compute the left child node select in bool array based on order map, feature and split_point_index

        Args:
            feature (int): which feature to split on
            split_point_index (int): choose which bucket
            sampled_indices (Union[List[int], None], optional): samples in original node.
                Defaults to None. None means all.

        Returns:
            np.ndarray: a 0/1 select array, shape (1 , sample number).
                1 means in left child, 0 otherwise.
        """
        if sampled_indices is None:
            length = self.ordermap_context.get_order_map_shape()[0]
            candidate = (
                self.ordermap_context.get_order_map()[:, feature] <= split_point_index
            )
        else:
            length = len(sampled_indices)
            candidate = (
                self.ordermap_context.get_order_map()[sampled_indices, feature]
                <= split_point_index
            )
        return candidate.astype(np.uint8).reshape(1, length)
