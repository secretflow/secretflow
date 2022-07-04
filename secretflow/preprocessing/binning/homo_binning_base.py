#!/usr/bin/env python3
# *_* coding: utf-8 *_*

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


import copy
import functools
import operator
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from secretflow.device import PYUObject, proxy
from secretflow.preprocessing.binning.kernels.base_binning import BaseBinning
from secretflow.preprocessing.binning.kernels.quantile_binning import QuantileBinning
from secretflow.preprocessing.binning.kernels.quantile_summaries import (
    QuantileSummaries,
)


@dataclass()
class SplitPointNode:
    """Dataclass of split point node

    Attributes:
        value: value of the split point
        min_value: min value of the split point
        max_value: nax value of the split point
        aim_rank: aim rank of the split point
        allow_error_rank: error tolerance on ranks
        error: create a new node if the difference is greater than error
        fixed: whether the split position converges
    """

    value: float
    min_value: float
    max_value: float
    aim_rank: int = -1
    allow_error_rank: int = 0
    error: float = 1e-4
    fixed: bool = False

    def create_right_new(self):
        """Search the right half"""
        value = (self.value + self.max_value) / 2
        if (
            np.fabs(value - self.value)
            <= (self.max_value - self.min_value) * self.error * 0.1
        ):
            self.fixed = True
            return self
        min_value = self.value
        return SplitPointNode(
            value, min_value, self.max_value, self.aim_rank, self.allow_error_rank
        )

    def create_left_new(self):
        """Search the left half"""
        value = (self.value + self.min_value) / 2
        if (
            np.fabs(value - self.value)
            <= (self.max_value - self.min_value) * self.error * 0.1
        ):
            self.fixed = True
            return self
        max_value = self.value
        return SplitPointNode(
            value, self.min_value, max_value, self.aim_rank, self.allow_error_rank
        )


@proxy(PYUObject)
class HomoBinningBase(BaseBinning):
    """Base class for horizontal federation binning

    Attributes:
        compress_thres: compression threshold. If the value is greater than the threshold, do compression
        error: error tolerance
        head_size: buffer size
        abnormal_list: list of anomaly features
        allow_duplicate: whether to allow duplicate bucket values
        aggregator: to aggregate values with aggregator
        max_values: a dict of max values for each features
        min_values: a dict of min values for each features
        total_count: total count
        columns: feature names
    """

    def __init__(
        self,
        bin_num: int = 10,
        bin_names: List[str] = [],
        bin_indexes: List[int] = [],
        compress_thres: int = 10000,
        error: float = 1e-4,
        head_size: int = 10000,
        allow_duplicate: bool = True,
        abnormal_list: List = None,
    ):
        super().__init__(
            bin_names=bin_names,
            bin_indexes=bin_indexes,
            bin_num=bin_num,
            abnormal_list=abnormal_list,
        )
        self.compress_thres = compress_thres
        self.error = error
        self.head_size = head_size
        self.abnormal_list = abnormal_list
        self.allow_duplicate = allow_duplicate
        self.max_values, self.min_values = None, None
        self.total_count = 0
        self.columns = []
        self.summary_dict = None
        self.missing_count = None
        self.query_points_dict = None
        self.missing_dict = {}
        self.split_num = None
        self.query_points = None

    def get_missing_count(self) -> Dict[str, int]:
        """statistics of missing count of all parties

        Returns:
            missing_count_dict: a dict store missing count of each features
        """
        missing_count_list = []
        columns = []
        for col, summary in self.summary_dict.items():
            columns.append(col)
            missing_count_list.append(summary.missing_count)
        return np.array(missing_count_list)

    def set_missing_dict(self, missing_count):
        for idx, col in enumerate(self.summary_dict.keys()):
            self.missing_dict[col] = missing_count[idx]

    def cal_summary_dict(self, data):
        self.summary_dict = QuantileBinning.feature_summary(
            data,
            compress_thres=self.compress_thres,
            head_size=self.head_size,
            error=self.error,
            bin_dict=self.bin_idx_name,
            abnormal_list=self.abnormal_list,
        )
        return self.summary_dict

    def init_query_points(
        self,
        split_num: int,
        error_rank: int = 1,
        need_first: bool = True,
        max_values=None,
        min_values=None,
        total_count=None,
    ) -> Dict[str, List[SplitPointNode]]:
        """
        query points initialize

        Args:
            split_num: how many buckets need to be split
            error_rank: error tolerance for rank
            need_first: whether splitPoint contains the minimum point.
            max_values: a dict store max values of each features
            min_values: a dict store min values of each features
            total_count: total count of
        """
        query_points_dict = {}
        self.split_num = split_num
        self.total_count = total_count
        for idx, col_name in enumerate(self.bin_names):
            max_value = max_values[col_name]
            min_value = min_values[col_name]
            sps = np.linspace(min_value, max_value, split_num)
            if not need_first:
                sps = sps[1:]

            split_point_array = [
                SplitPointNode(
                    sps[i],
                    min_value,
                    max_value,
                    error=self.error,
                    allow_error_rank=error_rank,
                )
                for i in range(len(sps))
            ]
            query_points_dict[col_name] = split_point_array
        self.query_points_dict = query_points_dict
        return query_points_dict

    def fit_split_points(self, data):
        pass

    def query_values(self):
        """Query what is the global rank for each current partition point
        Returns:
            global_rank: Dict
            eg: {col1: [g_rank1],
                 col2: [g_rank2]
            }

        """
        columns = self.summary_dict.keys()
        local_rank = []
        for col in columns:
            col_local_rank = self.query_table(
                self.summary_dict[col], self.query_points_dict[col]
            )
            local_rank.append(col_local_rank)
        return np.array(local_rank)

    def query_table(
        self,
        summary: Dict[str, QuantileSummaries],
        query_points: Dict[str, List[SplitPointNode]],
    ) -> np.array:
        """Query the rank of query_points in the local summary

        Args:
            summary: a dict store summary of each features
            query_points:{
                col1: [SplitPointNode,...,SplitPointNode],
                col2: [SplitPointNode,...,SplitPointNode],
                ...
            }
        """

        queries = [x.value for x in query_points]
        original_idx = np.argsort(np.argsort(queries))
        queries = np.sort(queries)
        ranks = summary.batch_query_value(queries)
        ranks = np.array(ranks)[original_idx]
        return np.array(ranks, dtype=int)

    def set_aim_rank(self):
        for col, split_point_array in self.query_points_dict.items():
            t_count = self.total_count - self.missing_dict[col]
            aim_ranks = [
                np.floor(x * t_count) for x in np.linspace(0, 1, self.split_num)
            ]
            aim_ranks = aim_ranks[1:]
            for idx, sp in enumerate(split_point_array):
                sp.aim_rank = aim_ranks[idx]

    def set_header_param(
        self,
        bin_names: List[str],
        bin_indexes: List[str],
        bin_idx_name: List,
        col_name_maps: Dict,
    ):
        self.bin_names = bin_names
        self.bin_idx_name = bin_idx_name
        self.bin_indexes = bin_indexes
        self.col_name_maps = col_name_maps

    def get_split_points_dict(self):
        return self.query_points_dict

    def renew_query_points(self, global_ranks: List):
        """Use to update query points

        Args:
            query_points: A list of split points for a column[splitNode0, splitNode1, ... , splitNodeN]

        Returns:
            List: A list after split
        """
        query_idx = 0
        for col, query_points in self.query_points_dict.items():
            new_array = []
            ranks = global_ranks[query_idx]
            for idx, node in enumerate(query_points):
                rank = ranks[idx]

                if node.fixed:
                    new_node = copy.deepcopy(node)
                elif rank - node.aim_rank > node.allow_error_rank:
                    new_node = node.create_left_new()
                elif node.aim_rank - rank > node.allow_error_rank:
                    new_node = node.create_right_new()
                else:
                    new_node = copy.deepcopy(node)
                    new_node.fixed = True
                new_node.last_rank = rank
                new_array.append(new_node)

            self.query_points_dict[col] = new_array
            query_idx += 1
        return self.check_converge()

    def check_converge(self) -> bool:
        """check convergence of federate binning

        Returns:
            bool : Returns convergence
        """

        def is_all_fixed(node_array):
            fix_array = [n.fixed for n in node_array]
            return functools.reduce(operator.and_, fix_array)

        fix_list = []
        for col, query_points in self.query_points_dict.items():
            fix_list.append(is_all_fixed(query_points))

        return functools.reduce(operator.and_, fix_list)

    def get_bin_result(self):
        bin_results = {}
        for col_name, sps in self.query_points_dict.items():
            sp = [x.value for x in sps]
            if not self.allow_duplicate:
                sp = sorted(set(sp))
                res = [sp[0] if np.fabs(sp[0]) > self.error else 0.0]
                last = sp[0]
                for v in sp[1:]:
                    if np.fabs(v) < self.error:
                        v = 0.0
                    if np.abs(v - last) > self.error:
                        res.append(v)
                    last = v
                sp = res
            bin_results[col_name] = sp
        return bin_results
