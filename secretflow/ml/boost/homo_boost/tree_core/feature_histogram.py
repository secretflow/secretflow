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


import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from operator import add, sub
from typing import Dict, List

import numpy
import numpy as np
import pandas

from secretflow.utils.errors import InvalidArgumentError


@dataclass()
class HistogramBag(object):
    """Histogram container

    Attributes:
        histogram: Histogram list calculated by calculate_histogram
        hid: histogram id
        p_hid: parent histogram id
    """

    histogram: List = None
    hid: int = -1
    p_hid: int = -1

    def binary_op(self, other, func: callable, inplace: bool = False):
        assert isinstance(
            other, HistogramBag
        ), f"Expect HistogramBag but got instance of {type(other)}"
        assert len(self.histogram) == len(
            other
        ), f"Expect two same length factors, but got {len(self.histogram)} and {len(other)}"

        histogram = self.histogram
        new_histogram = None
        if not inplace:
            new_histogram = copy.deepcopy(other)
            histogram = new_histogram.histogram
        for f_idx in range(len(self.histogram)):
            for hist_idx in range(len(self.histogram[f_idx])):
                # grad
                histogram[f_idx][hist_idx][0] = func(
                    self.histogram[f_idx][hist_idx][0], other[f_idx][hist_idx][0]
                )
                # hess
                histogram[f_idx][hist_idx][1] = func(
                    self.histogram[f_idx][hist_idx][1], other[f_idx][hist_idx][1]
                )
                # sample
                histogram[f_idx][hist_idx][2] = func(
                    self.histogram[f_idx][hist_idx][2], other[f_idx][hist_idx][2]
                )

        return self if inplace else new_histogram

    def __add__(self, other):
        return self.binary_op(other, add, inplace=False)

    def __sub__(self, other):
        return self.binary_op(other, sub, inplace=False)

    def __len__(self):
        return len(self.histogram)

    def __getitem__(self, item: int):
        return self.histogram[item]

    def __str__(self):
        return str(self.histogram)

    def __repr__(self):
        return str(self.histogram)


class FeatureHistogram:
    """Feature Histogram"""

    @staticmethod
    def _cal_histogram_once(
        data_frame,
        bin_split_points,
        valid_features,
        use_missing,
        grad_key,
        hess_key,
        thread_pool,
    ):
        if len(data_frame) == 0:
            f_histogram = FeatureHistogram._generate_empty_histogram(
                bin_split_points, valid_features, 1 if use_missing else 0
            )
        else:
            f_histogram = FeatureHistogram._node_calculate_histogram(
                data_frame,
                bin_split_points=bin_split_points,
                valid_features=valid_features,
                use_missing=use_missing,
                grad_key=grad_key,
                hess_key=hess_key,
                thread_pool=thread_pool,
            )
        return f_histogram

    @staticmethod
    def calculate_histogram(
        data_frame_list: List[pandas.DataFrame],
        bin_split_points: numpy.ndarray,
        valid_features: Dict = None,
        use_missing: bool = False,
        grad_key: str = "grad",
        hess_key: str = "hess",
        thread_pool: ThreadPoolExecutor = None,
    ):
        """
        Calculate histogram according to G and H
        histogram: [cols,[buckets,[sum_g,sum_h,count]]

        Args:
            data_frame_list: A list of data frame, which contain grad and hess
            bin_split_points: global split point dicts
            valid_features: valid feature names Dict[id:bool]
            use_missing: whether missing value participate in train
            grad_key: unique column name for grad value
            hess_key: unique column name for hess value
        Returns:
            node_histograms:一个List[histogram1, histogram2, ...]
        """
        node_histograms = []
        for data_frame in data_frame_list:
            node_histograms.append(
                FeatureHistogram._cal_histogram_once(
                    data_frame,
                    bin_split_points,
                    valid_features,
                    use_missing,
                    grad_key,
                    hess_key,
                    thread_pool,
                )
            )

        return node_histograms

    @staticmethod
    def _generate_empty_histogram(
        bin_split_points: Dict, valid_features: Dict, missing_bin: int
    ):
        """If data if empty, generate empty histogram
        Args:
            bin_split_points: global bin split points
            valid_features: Dict for valid features
            missing_bin: Num of missing bin
        Returns:
            feature_histogram_template: return empty histogram
        """
        feature_histogram_template = []
        for fid in range(len(bin_split_points)):
            # if is not valid features, skip generating
            if valid_features is not None and valid_features[fid] is False:
                feature_histogram_template.append([])
                continue
            else:
                # [0, 0, 0] -> [grad, hess, sample count]
                feature_histogram_template.append(
                    [[0, 0, 0] for j in range(len(bin_split_points[fid]) + missing_bin)]
                )

        # check feature num
        assert len(feature_histogram_template) == len(
            bin_split_points
        ), "Length of feature_histogram_template and bin_split_points not consistent"

        return feature_histogram_template

    @staticmethod
    def _cal_point_hist(data_slice):
        sum_mat = data_slice.sum(axis=0)
        sum_grad = sum_mat[1]
        sum_hess = sum_mat[2]
        sum_count = data_slice.shape[0]
        return [sum_grad, sum_hess, sum_count]

    @staticmethod
    def calculate_single_histogram(data: np.ndarray, bin_split_point: np.ndarray):
        f_histogram = []

        for bin_t in bin_split_point:
            f_histogram.append(
                FeatureHistogram._cal_point_hist(data[data[:, 0] < bin_t])
            )

        return f_histogram

    @staticmethod
    def _node_calculate_histogram(
        data_frame: pandas.DataFrame,
        bin_split_points: np.ndarray = None,
        valid_features: Dict = None,
        use_missing: bool = False,
        grad_key="grad",
        hess_key="hess",
        thread_pool=None,
    ):
        """function to calculate histogram on node

        Args:
            data_frame: data frame with grad and hess
            bin_split_points: global bin split point
            valid_features: valid feature names Dict[id:bool]
            use_missing: whether missing value participate in train
            grad_key: unique column name for grad value
            hess_key: unique column name for hess value

        Returns:
            single_histogram: histogram of this node
        """
        if thread_pool is None:
            thread_pool = ThreadPoolExecutor()
        single_histogram = []
        np_data = data_frame.to_numpy()
        if valid_features is None:
            raise InvalidArgumentError("valid can not be None")
        header = data_frame.columns.tolist()
        valid_features_list = [k for k, v in valid_features.items() if v]

        futures = {}
        for fid in range(len(bin_split_points)):
            futures[fid] = thread_pool.submit(
                FeatureHistogram._bin_hist,
                np_data,
                fid,
                grad_key,
                hess_key,
                valid_features_list,
                header,
                bin_split_points,
                use_missing,
            )

        for ret in futures.values():
            single_histogram.append(np.array(ret.result()))
        return single_histogram

    @staticmethod
    def _bin_hist(
        data,
        fid,
        grad_key,
        hess_key,
        valid_features_list,
        header,
        bin_split_points,
        use_missing,
    ):
        if fid in valid_features_list:
            t_data = data[:, [fid, header.index(grad_key), header.index(hess_key)]]
            f_histogram = FeatureHistogram.calculate_single_histogram(
                t_data, bin_split_points[fid]
            )
            if use_missing:
                miss_grad = t_data[1:].sum() - f_histogram[-1][0]
                miss_hess = t_data[2:].sum() - f_histogram[-1][1]
                miss_count = len(t_data) - f_histogram[-1][2]
                f_histogram.append([miss_grad, miss_hess, miss_count])
        else:
            f_histogram = []
        return f_histogram
