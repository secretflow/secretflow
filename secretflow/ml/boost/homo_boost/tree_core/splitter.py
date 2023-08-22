#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


import logging
from dataclasses import dataclass
from typing import Dict, List

from secretflow.ml.boost.homo_boost.tree_core.criterion import XgboostCriterion


@dataclass()
class SplitInfo:
    """Split Info
    Attributes:
        best_fid: best split on feature id
        best_bid: best split on bucket id
        sum_grad: sum of grad
        sum_hess: sum of hess
        gain: split gain
        missing_dir: which branch to go when encounting missing value default 1->right
        sample_count: num of sample after split
    """

    best_fid: int = None
    best_bid: int = None
    sum_grad: float = 0
    sum_hess: float = 0
    gain: float = None
    missing_dir: int = 1
    sample_count: int = -1

    def __str__(self):
        return (
            f"(fid:{self.best_fid}, bid:{self.best_bid}, "
            f"sum_grad:{self.sum_grad}, sum_hess:{self.sum_hess}, gain:{self.gain}, "
            f"missing_dir:{self.missing_dir}, sample_count:{self.sample_count})\n"
        )

    def __repr__(self):
        return self.__str__()


class Splitter(object):
    """Split Calculate Class
    Attributes:
        criterion_method: criterion method
        criterion_params: criterion parms, eg[l1: 0.1, l2: 0.2]
        min_impurity_split: minimum gain threshold of splitting
        min_sample_split: minimum sample split of splitting, default to 2
        min_leaf_node: minimum samples on node to split
        min_child_weight: minimum sum of hess after split
    """

    def __init__(
        self,
        criterion_method: str,
        criterion_params: List = [0, 0, 10],
        min_impurity_split: float = 1e-2,
        min_sample_split: int = 2,
        min_leaf_node: int = 1,
        min_child_weight: int = 1,
    ):
        if not isinstance(criterion_method, str):
            raise TypeError(
                "criterion_method type should be str, but %s find"
                % (type(criterion_method).__name__)
            )

        if criterion_method == "xgboost":
            if not criterion_params:
                self.criterion = XgboostCriterion()
            else:
                if type(criterion_params) is list:
                    try:
                        reg_lambda = float(criterion_params[0])
                        reg_alpha = float(criterion_params[1])
                        decimal = int(criterion_params[2])
                        self.criterion = XgboostCriterion(
                            reg_lambda=reg_lambda, reg_alpha=reg_alpha, decimal=decimal
                        )
                    except ValueError:
                        logging.warn(
                            "criterion_params' first criterion_params should be numeric"
                        )
                        self.criterion = XgboostCriterion()

        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight

    def _check_min_child_weight(self, l_h: float, r_h: float) -> bool:
        return l_h >= self.min_child_weight and r_h >= self.min_child_weight

    def _check_sample_num(self, l_cnt: int, r_cnt: int) -> bool:
        return l_cnt >= self.min_leaf_node and r_cnt >= self.min_leaf_node

    def find_split_once(
        self, histogram: List, valid_features: Dict, use_missing: bool
    ) -> SplitInfo:
        """Find best split info from histogram

        Args:
            histogram: a three-dimensional matrix store G,H,Count
            valid_features: valid feature names Dict[id:bool]
            use_missing: whether missing value participate in train
        Returns:
            SplitInfo: best split point info
        """
        best_fid = None
        best_gain = self.min_impurity_split
        best_bid = None
        best_sum_grad_l = None
        best_sum_hess_l = None
        best_sample_cnt = None
        missing_bin = 0
        if use_missing:
            missing_bin = 1

        # default to right
        missing_dir = 1

        for fid in range(len(histogram)):
            if valid_features[fid] is False:
                continue
            bin_num = len(histogram[fid])
            if bin_num == 0 + missing_bin:
                continue

            # The last bucket stores the sum of all nodes(cumsum from left)
            sum_grad = histogram[fid][bin_num - 1][0]
            sum_hess = histogram[fid][bin_num - 1][1]
            node_cnt = histogram[fid][bin_num - 1][2]
            if node_cnt < self.min_sample_split:
                break

            # The last bucket does not participate in the split point search, so bin_num-1
            for bid in range(bin_num - missing_bin - 1):
                # left gh
                sum_grad_l, sum_hess_l, node_cnt_l = histogram[fid][bid]

                # right gh
                sum_grad_r = sum_grad - sum_grad_l
                sum_hess_r = sum_hess - sum_hess_l
                node_cnt_r = node_cnt - node_cnt_l
                logging.debug(
                    f"split::fid={fid} bid={bid} sum_of_grad_l:{sum_grad_l},sum_of_hess_l={sum_hess_l} ,"
                    f"node_cnt_l={node_cnt_l}"
                )
                if self._check_sample_num(
                    node_cnt_l, node_cnt_r
                ) and self._check_min_child_weight(sum_hess_l, sum_hess_r):
                    gain = self.criterion.split_gain(
                        (sum_grad, sum_hess),
                        (sum_grad_l, sum_hess_l),
                        (sum_grad_r, sum_hess_r),
                    )
                    if gain > self.min_impurity_split and gain > best_gain:
                        best_gain = gain
                        best_fid = fid
                        best_bid = bid
                        best_sum_grad_l = sum_grad_l
                        best_sum_hess_l = sum_hess_l
                        best_sample_cnt = node_cnt_l
                        missing_dir = 1

                # handle missing value: dispatch to left sub tree
                if use_missing:
                    sum_grad_l += histogram[fid][-1][0] - histogram[fid][-2][0]
                    sum_hess_l += histogram[fid][-1][1] - histogram[fid][-2][1]
                    node_cnt_l += histogram[fid][-1][2] - histogram[fid][-2][2]

                    sum_grad_r -= histogram[fid][-1][0] - histogram[fid][-2][0]
                    sum_hess_r -= histogram[fid][-1][1] - histogram[fid][-2][1]
                    node_cnt_r -= histogram[fid][-1][2] - histogram[fid][-2][2]

                    # If the left side is gain more, point missing_DIR to the left
                    # and update the optimal split information
                    if self._check_sample_num(
                        node_cnt_l, node_cnt_r
                    ) and self._check_min_child_weight(sum_hess_l, sum_hess_r):
                        gain = self.criterion.split_gain(
                            (sum_grad, sum_hess),
                            (sum_grad_l, sum_hess_l),
                            (sum_grad_r, sum_hess_r),
                        )
                        if gain > self.min_impurity_split and gain > best_gain:
                            best_gain = gain
                            best_fid = fid
                            best_bid = bid
                            best_sum_grad_l = sum_grad_l
                            best_sum_hess_l = sum_hess_l
                            missing_dir = -1
                            best_sample_cnt = node_cnt_l

        splitinfo = SplitInfo(
            best_fid=best_fid,
            best_bid=best_bid,
            gain=best_gain,
            sum_grad=best_sum_grad_l,
            sum_hess=best_sum_hess_l,
            missing_dir=missing_dir,
            sample_count=best_sample_cnt,
        )
        logging.debug(f"splitInfo = {splitinfo}")
        return splitinfo

    def find_split(
        self, histograms: List, valid_features: Dict, use_missing: bool = False
    ) -> List[SplitInfo]:
        """查找最优分裂点
        Args:
            histograms: a list of histogram
            valid_features: valid feature names Dict[id:bool]
            use_missing: whether missing value participate in train

        Return:
            tree_node_splitinfo: best split info on each node
        """
        tree_node_splitinfo = []
        for histogram in histograms:
            splitinfo_node = self.find_split_once(
                histogram, valid_features, use_missing
            )
            tree_node_splitinfo.append(splitinfo_node)

        return tree_node_splitinfo

    def node_gain(self, grad: float, hess: float) -> float:
        return self.criterion.node_gain(grad, hess)

    def node_weight(self, grad: float, hess: float) -> float:
        return self.criterion.node_weight(grad, hess)

    def split_gain(
        self,
        sum_grad: float,
        sum_hess: float,
        sum_grad_l: float,
        sum_hess_l: float,
        sum_grad_r: float,
        sum_hess_r: float,
    ) -> float:
        return self.criterion.split_gain(
            [sum_grad, sum_hess], [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r]
        )
