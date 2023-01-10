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
        samples = len(sorted_x)
        remained_count = samples
        assert remained_count > 0, 'can not qcut empty x'

        value_category = list()
        last_value = None

        split_points = list()
        idx = 0
        expected_idx = math.ceil(remained_count / self.buckets)
        fast_skip = False
        while idx < samples:
            v = sorted_x[idx]
            value_diff = v != last_value
            if not fast_skip and value_diff:
                if len(value_category) <= self.buckets:
                    value_category.append(v)
                else:
                    fast_skip = True
                last_value = v

            if idx >= expected_idx and value_diff:
                split_points.append(v)
                if len(split_points) == self.buckets - 1:
                    break
                remained_count = samples - idx
                expected_bin_count = math.ceil(
                    remained_count / (self.buckets - len(split_points))
                )
                expected_idx = idx + expected_bin_count
                last_value = v

            if not fast_skip or idx >= expected_idx:
                idx += 1
            else:
                idx = expected_idx

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

    def _build_maps(self, x: np.ndarray):
        '''
        split features into buckets and build maps use in train.
        '''
        # order_map: record sample belong to which bucket of all features.
        self.order_map = np.zeros((x.shape[0], x.shape[1]), dtype=np.int8, order='F')
        # split_points: bucket split points for all features.
        self.split_points = []
        # feature_buckets: how many buckets in each feature.
        self.feature_buckets = []
        # features: how many features in dataset.
        self.features = x.shape[1]
        for f in range(x.shape[1]):
            bins, split_point = self._qcut(x[:, f])
            self.order_map[:, f] = bins
            total_buckets = len(split_point) + 1

            self.feature_buckets.append(total_buckets)
            # last bucket is split all samples into left child.
            # using infinity to simulation xgboost pruning.
            split_point.append(float('inf'))
            self.split_points.append(split_point)

    def build_bucket_map(self, start: int, length: int) -> np.ndarray:
        '''
        Build bucket_map fragment base on order_map.
        '''
        end = start + length
        assert end <= self.order_map.shape[0]

        total_buckets = sum(self.feature_buckets)
        buckets_map = np.zeros((length, total_buckets), dtype=np.int8)
        feature_bucket_pos = 0
        for f in range(self.order_map.shape[1]):
            feature_bucket = self.feature_buckets[f]
            for bucket in range(feature_bucket):
                bin_idx = np.flatnonzero(self.order_map[start:end, f] == bucket)
                for b in range(bucket, feature_bucket):
                    buckets_map[bin_idx, feature_bucket_pos + b] = 1
            feature_bucket_pos += feature_bucket
        return buckets_map

    def global_setup(self, x: np.ndarray, buckets: int, seed: int):
        '''
        Set up global context.
        '''
        np.random.seed(seed)
        x = np.array(x, order='F')
        # max buckets in each feature.
        self.buckets = buckets
        self._build_maps(x)

    def update_buckets_count(
        self, buckets_count: List[Tuple[int, int]], buckets_choices: np.ndarray
    ) -> np.ndarray:
        '''
        save how many buckets in each partition's features.
        and add offset for buckets_choices if colsample < 1
        '''
        self.buckets_count = [b[0] for b in buckets_count]
        if buckets_choices is not None:
            return buckets_choices + sum([b[1] for b in buckets_count[: self.work_idx]])
        return None

    def tree_setup(self, colsample: float) -> Tuple[np.ndarray, Tuple[int, int]]:
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
            chosen_buckets = 0
            total_buckets = 0
            for f_idx, f_buckets_size in enumerate(self.feature_buckets):
                if f_idx in self.col_choices:
                    buckets_choices.extend(
                        range(total_buckets, total_buckets + f_buckets_size)
                    )
                    chosen_buckets += f_buckets_size
                total_buckets += f_buckets_size

            return np.array(buckets_choices, dtype=np.int32), (
                chosen_buckets,
                total_buckets,
            )
        else:
            self.col_choices = None
            total_buckets = sum(self.feature_buckets)
            return None, (total_buckets, total_buckets)

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
