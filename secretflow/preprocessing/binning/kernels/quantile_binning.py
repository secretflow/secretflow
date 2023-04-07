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


import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from secretflow.preprocessing.binning.kernels.base_binning import BaseBinning
from secretflow.preprocessing.binning.kernels.quantile_summaries import (
    QuantileSummaries,
)


class QuantileBinning(BaseBinning):
    """Use QuantileSummary algorithm for constant frequency binning

    Attributes:
        bin_num: the num of buckets
        compress_thres: if size of summary greater than compress_thres, do compress operation
        cols_dict: mapping of value to index. {key: col_name , value: index}.
        head_size: buffer size
        error: 0 <= error < 1 default: 0.001,error tolerance, floor((p - 2 * error) * N) <= rank(x) <= ceil((p + 2 * error) * N)
        abnormal_list: list of anomaly features.
        summary_dict: a dict store summary of each features
        col_name_maps: a dict store column index to name
        bin_idx_name: a dict store index to name
        allow_duplicate: Whether duplication is allowed
    """

    def __init__(
        self,
        bin_num: int = 10,
        compress_thres: int = 10000,
        head_size: int = 10000,
        error: float = 1e-4,
        bin_indexes: List[int] = [],
        bin_names: List[str] = [],
        local_only: bool = False,
        abnormal_list: List[str] = None,
        allow_duplicate: bool = False,
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
        self.local_only = local_only
        self.abnormal_list = abnormal_list
        self.summary_dict = None
        self.col_name_maps = {}
        self.bin_idx_name = {}
        self.allow_duplicate = allow_duplicate

    def fit_split_points(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        calculate bin split points base on QuantileSummary algorithm

        Args:
            data_frame: input data

        Returns:
            bin_result: bin result returned as dataframe
        """
        self.header = data_frame.columns.tolist()
        (
            self.bin_names,
            self.bin_indexes,
            self.bin_idx_name,
            self.col_name_maps,
        ) = self._setup_header_param(self.header, self.bin_names, self.bin_indexes)
        logging.debug(f"Header length: {len(self.header)}")

        self._fit_split_point(data_frame)
        return pd.DataFrame.from_dict(self.bin_results)

    def _fit_split_point(self, data_frame: pd.DataFrame):
        def _cal_split_point(data: np.ndarray):
            split_point = []
            for ab_item in self.abnormal_list:
                data = data[data != ab_item]
            bin_list = (
                np.linspace(0, len(data), self.bin_num + 1).round()[1:].astype(int)
            )
            sorted_data = np.sort(data)
            for bin_t in bin_list:
                rank_t = sorted_data[bin_t - 1]
                split_point.append(rank_t)
            return split_point

        header = data_frame.columns.tolist()
        for col_name in header:
            split_point = _cal_split_point(data_frame[col_name].to_numpy())
            self.bin_results[col_name] = list(set(split_point))

    @staticmethod
    def feature_summary(
        data_frame: pd.DataFrame,
        compress_thres: int,
        head_size: int,
        error: float,
        bin_dict: Dict[str, int],
        abnormal_list: List[str],
    ) -> Dict:
        """
        calculate summary

        Args:
            data_frame: pandas.DataFrame, input data
            compress_thres: int,
            head_size: int, buffer size, when
            error: float, error tolerance
            bin_dict: a dict store col name to index
            abnormal_list: list of anomaly features
        """
        summary_dict = {}
        summary_param = {
            'compress_thres': compress_thres,
            'head_size': head_size,
            'error': error,
            'abnormal_list': abnormal_list,
        }
        for bin_index, bin_name in bin_dict.items():
            quantile_summaries = QuantileSummaries(**summary_param)
            summary_dict[bin_name] = quantile_summaries

        for col_name, summary in summary_dict.items():
            summary.fast_init(data_frame[col_name].to_numpy())

        result = {}
        for features_name, summary_obj in summary_dict.items():
            summary_obj.compress()
            result[features_name] = summary_obj

        return result
