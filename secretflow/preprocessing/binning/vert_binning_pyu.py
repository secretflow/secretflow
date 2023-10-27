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

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from secretflow.device import PYUObject, proxy


@proxy(PYUObject)
class VertBinningPyuWorker:
    """
    PYU functions for binning
    Attributes: see VertBinning
    """

    def __init__(
        self,
        data: pd.DataFrame,
        binning_method: str,
        bin_num: int,
        bin_names: List[str],
    ):
        data_columns = data.columns
        assert isinstance(
            bin_names, list
        ), f"bin names should be a list of string but got {type(bin_names)}"
        assert np.isin(
            bin_names, data_columns
        ).all(), (
            f"bin_names[{bin_names}] not exist in input data columns[{data_columns}]"
        )
        self.data = data
        self.bin_names = bin_names
        self.bin_num = bin_num
        self.binning_method = binning_method

    def _build_feature_bin(
        self, f_data: pd.DataFrame
    ) -> Tuple[List[np.ndarray], Union[np.ndarray, List[str]], np.ndarray]:
        '''
        split one feature column into {bin_num} bins.

        Attributes:
            f_data: feature column to be split.

        Return:
            First: sample indices for each bins.
            Second: split points for number column (np.array) or
                    categories for string column (List[str])
            Third: sample indices for np.nan values.
        '''
        if f_data.dtype == np.dtype(object):
            # for string type col, split into bins by categories.
            categories = {d for d in f_data if not pd.isna(d)}
            for c in categories:
                assert isinstance(
                    c, str
                ), f"only support str if dtype == np.obj, but got {type(c)}"
            split_points = sorted(list(categories))
            bin_indices = list()
            for b in split_points:
                bin_indices.append(np.flatnonzero(f_data == b))
            return bin_indices, split_points, np.flatnonzero(pd.isna(f_data))
        else:
            # for number type col, first binning by pd.qcut.
            bin_num = self.bin_num

            if self.binning_method == "quantile":
                bins, split_points = pd.qcut(
                    f_data, bin_num, labels=False, duplicates='drop', retbins=True
                )
            elif self.binning_method == "eq_range":
                bins, split_points = pd.cut(
                    f_data, bin_num, labels=False, duplicates='drop', retbins=True
                )
            else:
                raise ValueError(f"binning_method {self.binning_method} not supported")

            bin_indices = list()
            assert split_points.size >= 2, f"split_points.size {split_points.size}"
            empty_bins = [0, split_points.size - 1]
            for b in range(split_points.size - 1):
                bin = np.flatnonzero(bins == b)
                if bin.size == 0:
                    # Then, remove empty bin in pd.qcut's result.
                    empty_bins.append(b + 1)
                else:
                    bin_indices.append(bin)

            return (
                bin_indices,
                # remove start/end value & empty bins in pd.qcut's range result.
                # remain only left-open right-close split points
                np.delete(split_points, empty_bins),
                np.flatnonzero(pd.isna(f_data)),
            )

    def _build_feature_bins(
        self, data: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[Union[np.ndarray, List[str]]], List[np.ndarray]]:
        '''
        split all columns into {bin_num} bins.
        Attributes:
            data: dataset to be split.

        Return:
            First: sample indices for each bins in all features.
            Second: split points for number column (np.array) or
                    categories for string column (List[str]) for all features.
            Third: sample indices for np.nan values in all features.
        '''
        ret_bins_idx = list()
        ret_points = list()
        ret_else_bins = list()
        assert isinstance(data, pd.DataFrame), type(data)
        for f_name in self.bin_names:
            f_data = data.loc[:, f_name]
            bin_idx, split_point, else_bin = self._build_feature_bin(f_data)
            if isinstance(split_point, list):
                # use List[str] for string column
                # split_point means categories, so length of it need equal to bin_idx
                assert len(bin_idx) == len(split_point), (
                    f"len(bin_idx) {len(bin_idx)},"
                    f" len(split_point) {len(split_point)}"
                )
            else:
                # use np.array for number column
                # split_point contain left-open right-close split points between each bins.
                # so length of it need equal to len(bin_idx) - 1
                assert len(bin_idx) == split_point.size + 1, (
                    f"len(bin_idx) {len(bin_idx)},"
                    f" split_point.size {split_point.size}"
                )
            ret_bins_idx += bin_idx
            ret_points.append(split_point)
            ret_else_bins.append(else_bin)

        return ret_bins_idx, ret_points, ret_else_bins

    def _build_report_dict(
        self,
        f_name: str,
        split_points: Union[np.ndarray, List[str]],
        total_counts: List[int],
        else_counts: int,
    ) -> Dict:
        '''
        build report dict for one feature.
        Attributes:
            f_name: feature name.
            split_points: see _build_feature_bin.
            total_counts: total samples in each bins.
            else_counts: total samples for np.nan values.

        Return:
            Dict report.
        '''
        ret = dict()
        ret['name'] = f_name

        f_bin_size = 0
        if isinstance(split_points, list):
            ret['type'] = "string"
            ret['categories'] = split_points
            f_bin_size = len(split_points)
        else:
            ret['type'] = "numeric"
            ret['split_points'] = list(split_points)
            f_bin_size = split_points.size + 1

        ret['total_counts'] = list()
        for i in range(len(total_counts)):
            ret['total_counts'].append(total_counts[i])

        ret['else_counts'] = else_counts

        ret['filling_values'] = [*range(f_bin_size)]

        ret['else_filling_value'] = -1
        return ret

    def _build_report(
        self,
        split_points: List[Union[np.ndarray, List[str]]],
        total_counts: List[int],
        else_counts: List[int],
    ) -> Dict:
        '''
        Attributes:
            split_points: see _build_feature_bin.
            total_counts: total samples all features' bins.
            else_counts: np.nan samples in all features.

        Return:
            Dict report
        '''
        assert len(else_counts) == len(self.bin_names), (
            f"len(else_counts) {len(else_counts)},"
            f" len(self.bin_names) {len(self.bin_names)}"
        )
        assert len(split_points) == len(self.bin_names), (
            f"len(split_points) {len(split_points)},"
            f" len(self.bin_names) {len(self.bin_names)}"
        )

        pos = 0
        variables = list()
        for f_idx in range(len(split_points)):
            split_point = split_points[f_idx]
            f_bin_size = 0
            if isinstance(split_point, list):
                f_bin_size = len(split_point)
            else:
                f_bin_size = split_point.size + 1

            variables.append(
                self._build_report_dict(
                    self.bin_names[f_idx],
                    split_points[f_idx],
                    total_counts[pos : pos + f_bin_size],
                    else_counts[f_idx],
                )
            )
            pos += f_bin_size

        assert len(variables) == len(self.bin_names), (
            f"len(variables) {len(variables)}, "
            f"len(self.bin_names) {len(self.bin_names)}"
        )
        return {"variables": variables}

    def bin_feature_and_produce_rules(self) -> Dict:
        '''
        bin all the features in self.data and return rules
        '''
        bins_idx, split_points, else_bins = self._build_feature_bins(self.data)

        total_counts = [b.size for b in bins_idx]
        else_counts = [b.size for b in else_bins]
        return self._build_report(split_points, total_counts, else_counts)
