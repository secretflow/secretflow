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
from typing import List, Tuple, Union

import numpy as np

from .order_map_context import OrderMapContext


class Splitter:
    def __init__(self, idx: int):
        self.worker_index = idx
        self.order_map_context = OrderMapContext()

    def get_features(self) -> int:
        return self.order_map_context.get_features()

    def get_feature_buckets(self) -> List[int]:
        return self.order_map_context.get_feature_buckets()

    def get_feature_bucket_at(self, index: int) -> int:
        return self.order_map_context.get_feature_bucket_at(index)

    def get_order_map(self) -> np.ndarray:
        return self.order_map_context.get_order_map()

    def get_order_map_shape(self) -> Tuple[int, int]:
        return self.order_map_context.get_order_map_shape()

    def build_maps(self, x: np.ndarray, buckets: int):
        self.order_map_context.build_maps(x, buckets)

    def get_split_points(self) -> List[List[int]]:
        return self.order_map_context.get_split_points()

    def get_col_choices(self) -> List[int]:
        return self.col_choices

    def set_up_col_choices(self, colsample: float) -> Tuple[np.ndarray, int]:
        if colsample < 1:
            choices = math.ceil(self.get_features() * colsample)
            self.col_choices = np.sort(
                np.random.choice(self.get_features(), choices, replace=False)
            )

            buckets_count = 0
            for f_idx, f_buckets_size in enumerate(self.get_feature_buckets()):
                if f_idx in self.col_choices:
                    buckets_count += f_buckets_size

            return self.col_choices, buckets_count
        else:
            self.col_choices = None
            return None, sum(self.get_feature_buckets())

    def set_buckets_count(self, buckets_count: List[int]) -> None:
        """
        save how many buckets in each partition's all features.
        """
        self.buckets_count = buckets_count

    def find_split_bucket(self, split_bucket: int) -> int:
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

    def get_split_feature(self, split_bucket: int) -> Tuple[int, int]:
        """
        find split bucket is belong to which feature.
        """
        pre_end_pos = 0
        for f_idx in range(len(self.get_feature_buckets())):
            if self.col_choices is not None and f_idx not in self.col_choices:
                continue
            current_end_pod = pre_end_pos + self.get_feature_bucket_at(f_idx)
            if split_bucket < current_end_pod:
                return f_idx, split_bucket - pre_end_pos
            pre_end_pos += self.get_feature_bucket_at(f_idx)
        assert False, "should not be here, _get_split_feature"

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
            length = self.get_order_map_shape()[0]
            candidate = self.get_order_map()[:, feature] <= split_point_index
        else:
            length = len(sampled_indices)
            candidate = (
                self.get_order_map()[sampled_indices, feature] <= split_point_index
            )
        return candidate.astype(np.int8).reshape(1, length)
