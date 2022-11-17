# Copyright 2022 Ant Group Co., Ltd.
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
from typing import Tuple, List
import numpy as np
from .xgb_tree import XgbTree
from secretflow.device import PYUObject, proxy


@proxy(PYUObject)
class XgbTreeWorker:
    '''
    use in XGB model.
    do some compute works that only use one partition' dataset.
    '''

    def __init__(self, idx: int) -> None:
        self.work_idx = idx

    def predict_weight_select(self, x: np.ndarray, tree: XgbTree) -> np.ndarray:
        '''
        computer leaf nodes' sample selects known by this partition.

        Args:
            x: dataset from this partition.
            tree: tree model store by this partition.

        Return:
            leaf nodes' selects
        '''
        x = x if isinstance(x, np.ndarray) else np.array(x)
        split_nodes = len(tree.split_features)

        select = np.zeros((x.shape[0], split_nodes + 1), dtype=np.int8)
        # should parallel in c++
        for r in range(x.shape[0]):
            row = x[r, :]
            idxs = list()
            idxs.append(0)
            while len(idxs):
                idx = idxs.pop(0)
                if idx < split_nodes:
                    f = tree.split_features[idx]
                    v = tree.split_values[idx]
                    if f == -1:
                        # if node split by others partition's feature
                        # mark all split paths in tree.
                        idxs.append(idx * 2 + 1)
                        idxs.append(idx * 2 + 2)
                    else:
                        # if node split by this partition's feature
                        # mark the clearly split path in tree.
                        if row[f] < v:
                            idxs.append(idx * 2 + 1)
                        else:
                            idxs.append(idx * 2 + 2)
                else:
                    leaf_idx = idx - split_nodes
                    select[r, leaf_idx] = 1

        return select

    def _qcut(self, x: np.ndarray) -> Tuple[np.ndarray, List]:
        sorted_x = np.sort(x, axis=0)
        remained_count = len(sorted_x)
        assert remained_count > 0, 'can not qcut empty x'

        value_category = list()
        last_value = None

        split_points = list()
        expected_bin_count = math.ceil(remained_count / self.buckets)
        current_bin_count = 0
        for v in sorted_x:
            if v != last_value:
                if len(value_category) <= self.buckets:
                    value_category.append(v)

                if current_bin_count >= expected_bin_count:
                    split_points.append(v)
                    if len(split_points) == self.buckets - 1:
                        break
                    remained_count -= current_bin_count
                    expected_bin_count = math.ceil(
                        remained_count / (self.buckets - len(split_points))
                    )
                    current_bin_count = 0

                last_value = v
            current_bin_count += 1

        if len(value_category) <= self.buckets:
            # full dataset category count <= buckets
            # use category as split point.
            split_points = value_category[1:]
        elif split_points[-1] != sorted_x[-1]:
            # add max sample value into split_points like xgboost.
            split_points.append(sorted_x[-1])

        split_points = list(map(float, split_points))

        def upper_bound_bin(x: float):
            count = len(split_points)
            pos = 0
            while count > 0:
                step = math.floor(count / 2)
                v = split_points[pos + step]
                if x == v:
                    return pos + step + 1
                elif x > v:
                    pos = pos + step + 1
                    count -= step + 1
                else:
                    count = step
            return pos

        bins = np.vectorize(upper_bound_bin)(x)

        return bins, split_points

    def build_maps(self, x: np.ndarray) -> np.ndarray:
        '''
        split features into buckets and build maps use in train.

        Args:
            x: dataset from this partition.

        Return:
            leaf nodes' selects
        '''
        # order_map: record sample belong to which bucket of all features.
        self.order_map = np.zeros((x.shape[0], x.shape[1]), dtype=np.int8)
        # split_points: bucket split points for all features.
        self.split_points = []
        # feature_buckets: how many buckets in each feature.
        self.feature_buckets = []
        # features: how many features in dataset.
        self.features = x.shape[1]
        # buckets_map: a sparse 0-1 array use in compute the gradient sums.
        buckets_map = np.zeros((x.shape[0], 0), dtype=np.int8)
        for f in range(x.shape[1]):
            bins, split_point = self._qcut(x[:, f])
            self.order_map[:, f] = bins
            total_buckets = len(split_point) + 1
            f_buckets_map = np.zeros((x.shape[0], total_buckets), dtype=np.int8)

            sum_bin_idx = np.array([], dtype=np.int64)
            for b in range(total_buckets):
                bin_idx = np.flatnonzero(bins == b)
                sum_bin_idx = np.concatenate((sum_bin_idx, bin_idx), axis=None)
                f_buckets_map[sum_bin_idx, b] = 1

            buckets_map = np.concatenate((buckets_map, f_buckets_map), axis=1)
            self.feature_buckets.append(total_buckets)
            # last bucket is split all samples into left child.
            # using infinity to simulation xgboost pruning.
            split_point.append(float('inf'))
            self.split_points.append(split_point)

        return buckets_map

    def global_setup(self, x: np.ndarray, buckets: int, seed: int) -> np.ndarray:
        '''
        Set up global context.
        '''
        np.random.seed(seed)
        x = x if isinstance(x, np.ndarray) else np.array(x)
        self.buckets = buckets
        buckets_map = self.build_maps(x)
        return buckets_map

    def update_buckets_count(self, buckets_count: List[int]) -> None:
        '''
        save how many buckets in each partition's all features.
        '''
        self.buckets_count = buckets_count

    def tree_setup(self, colsample: float) -> Tuple[np.ndarray, int]:
        '''
        Set up tree context and do col sample if colsample < 1
        '''
        self.tree = XgbTree()
        if colsample < 1:
            choices = math.ceil(self.features * colsample)
            self.col_choices = np.sort(
                np.random.choice(self.features, choices, replace=False)
            )

            buckets_choices = []
            buckets_count = 0
            buckets_start = 0
            for f_idx, f_buckets_size in enumerate(self.feature_buckets):
                if f_idx in self.col_choices:
                    buckets_choices.extend(
                        range(buckets_start, buckets_start + f_buckets_size)
                    )
                    buckets_count += f_buckets_size
                buckets_start += f_buckets_size

            return np.array(buckets_choices, dtype=np.int32), buckets_count
        else:
            self.col_choices = None
            return None, sum(self.feature_buckets)

    def tree_finish(self) -> XgbTree:
        return self.tree

    def _find_split_bucket(self, split_bucket: int) -> int:
        '''
        check if this partition contains split bucket.
        '''
        pre_end_pos = 0
        for work_idx in range(len(self.buckets_count)):
            current_end_pod = pre_end_pos + self.buckets_count[work_idx]
            if split_bucket < current_end_pod:
                if work_idx == self.work_idx:
                    # split bucket is inside this partition's feature
                    return split_bucket - pre_end_pos
                else:
                    # split bucket is from other partition.
                    return -1
            pre_end_pos += self.buckets_count[work_idx]
        assert False, "should not be here, _is_primary_split"

    def _get_split_feature(self, split_bucket: int) -> Tuple[int, int]:
        '''
        find split bucket is belong to which feature.
        '''
        pre_end_pos = 0
        for f_idx in range(len(self.feature_buckets)):
            if self.col_choices is not None and f_idx not in self.col_choices:
                continue
            current_end_pod = pre_end_pos + self.feature_buckets[f_idx]
            if split_bucket < current_end_pod:
                return f_idx, split_bucket - pre_end_pos
            pre_end_pos += self.feature_buckets[f_idx]
        assert False, "should not be here, _get_split_feature"

    def do_split(self, split_buckets: List[int]) -> List[np.ndarray]:
        '''
        record split info and generate next level's left children select.
        '''
        lchild_selects = []
        for s in split_buckets:
            s = self._find_split_bucket(s)
            if s != -1:
                feature, split_point_idx = self._get_split_feature(s)
                self.tree.insert_split_node(
                    feature, self.split_points[feature][split_point_idx]
                )
                # lchild' select
                ls = (
                    (self.order_map[:, feature] <= split_point_idx)
                    .astype(np.int8)
                    .reshape(1, self.order_map.shape[0])
                )
                lchild_selects.append(ls)
            else:
                self.tree.insert_split_node(-1, float("inf"))
                lchild_selects.append(np.array([]))

        return lchild_selects
