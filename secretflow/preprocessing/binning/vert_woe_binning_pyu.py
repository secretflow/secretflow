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
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

from secretflow.device import PYUObject, proxy

from .kernels.chi_merge import apply_chimerge, update_split_points


@proxy(PYUObject)
class VertWoeBinningPyuWorker:
    """
    PYU functions for woe binning
    Attributes: see VertWoeBinning
    """

    def __init__(
        self,
        data: pd.DataFrame,
        binning_method: str,
        bin_num: int,
        bin_names: List[str],
        label_name: str,
        positive_label: str,
        chimerge_init_bins: int,
        chimerge_target_bins: int,
        chimerge_target_pvalue: float,
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
        self.bin_names = bin_names
        if label_name:
            assert np.isin(
                label_name, data_columns
            ).all(), f"label_name[{label_name}] not exist in input data columns[{data_columns}]"
            self.label_name = label_name
            self.positive_label = positive_label
        else:
            self.label_name = ""
        self.bin_num = bin_num
        self.binning_method = binning_method
        self.chimerge_init_bins = chimerge_init_bins
        self.chimerge_target_bins = chimerge_target_bins
        self.chimerge_target_chi = chi2.ppf(1 - chimerge_target_pvalue, df=1)
        # iv results
        self.iv_results = []

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
            bin_num = (
                self.bin_num
                if (
                    self.binning_method == "quantile"
                    or self.binning_method == "eq_range"
                )
                else self.chimerge_init_bins
            )
            if self.binning_method == "eq_range":
                bins, split_points = pd.cut(
                    f_data, bin_num, labels=False, duplicates='drop', retbins=True
                )
            else:
                bins, split_points = pd.qcut(
                    f_data, bin_num, labels=False, duplicates='drop', retbins=True
                )
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

    def _get_label(self, data: pd.DataFrame) -> np.array:
        '''
        Binarize label column.
        Attributes:
            data: input label column.

        Return:
            binarized label, 1 for positive, 0 for negative.
        '''
        assert isinstance(data, pd.DataFrame), type(data)
        raw_label = data.loc[:, self.label_name]

        if raw_label.dtype == np.dtype(object):
            assert isinstance(
                raw_label[0], str
            ), f"only support str if dtype == np.obj, but got {type(raw_label[0])}"
            return np.array((raw_label == self.positive_label)).astype(np.float32)
        else:
            positive_value = float(self.positive_label)
            return np.array((raw_label == positive_value)).astype(np.float32)

    def _build_iv_info_dict(
        self,
        f_name: str,
        ivs: List[float],
        else_iv: float,
    ) -> Dict:
        """
        build iv info dict

        Args:
            f_name (str): feature name.
            ivs (List[float]): bin ivs.
            else_iv (float): else ivs.

        Returns:
            Dict:  with the following infomation:
            {
                "name": str, #feature name
                "ivs": list[float], #iv values for each bins, not safe to share with workers in any case.
                "else_iv": float, #iv for nan values, may share to with workers
                "feature_iv": float, #sum of bin_ivs, safe to share with workers when bin num > 2.
            }
        """
        ret = dict()
        ret['name'] = f_name
        ret['ivs'] = ivs
        ret['else_iv'] = else_iv
        ret['feature_iv'] = sum(ivs)
        return ret

    def _build_report_dict(
        self,
        woes: List[float],
        f_name: str,
        split_points: Union[np.ndarray, List[str]],
        else_woe: float,
        total_counts: List[int],
        else_counts: int,
    ) -> Dict:
        '''
        build report dict for one feature.
        Attributes:
            woes: woe values for each bins in feature.
            f_name: feature name.
            split_points: see _build_feature_bin.
            else_woe: woe for np.nan values in feature.
            total_counts: total samples in each bins.
            else_counts: total samples for np.nan values.

        Return:
            Dict report.
        '''
        ret = dict()
        ret['name'] = f_name
        if isinstance(split_points, list):
            ret['type'] = "string"
            ret['categories'] = split_points
        else:
            ret['type'] = "numeric"
            ret['split_points'] = list(split_points)

        ret['filling_values'] = list()
        ret['total_counts'] = list()
        assert len(total_counts) == len(woes), (
            f"len(total_counts) {len(total_counts)}," f" len(woes) {len(woes)}"
        )
        for i in range(len(woes)):
            ret['total_counts'].append(total_counts[i])
            ret['filling_values'].append(woes[i])

        ret['else_filling_value'] = else_woe
        ret['else_counts'] = else_counts

        return ret

    def _calc_bin_woe_iv(
        self, bin_total: int, bin_positives: int
    ) -> Tuple[float, float]:
        '''
        calculate woe/iv for one bin.
        Attributes:
            bin_total: total samples in bin.
            bin_positives: positive samples in bin.

        Return:
            Tuple[woe, iv]
        '''
        total_labels = self.total_labels
        total_positives = self.total_positives
        total_negatives = total_labels - total_positives
        assert (
            total_positives > 0 and total_negatives > 0
        ), f"total_positives {total_positives}, total_negatives {total_negatives}"

        bin_negatives = bin_total - bin_positives

        positive_distrib = 0
        negative_distrib = 0

        if bin_negatives == 0 or bin_positives == 0:
            positive_distrib = (bin_positives + 0.5) / total_positives
            negative_distrib = (bin_negatives + 0.5) / total_negatives
        else:
            positive_distrib = bin_positives * 1.0 / total_positives
            negative_distrib = bin_negatives * 1.0 / total_negatives

        woe = math.log(positive_distrib / negative_distrib)
        iv = (positive_distrib - negative_distrib) * woe
        return (woe, iv)

    def accumulate_iv_info(
        self,
        ivs: List[float],
        bin_names: List[str],
        else_ivs: List[float],
        split_point_sizes: List[int],
    ):
        """accumulate iv info in results.

        Args:
            ivs (List[float]): bin ivs
            bin_names (List[str]): bin names
            else_iv (List[float]): nan iv.
            split_point_sizes (List[int]): _description_

        """
        pos = 0
        for f_idx in range(len(bin_names)):
            f_bin_size = split_point_sizes[f_idx]
            self.iv_results.append(
                self._build_iv_info_dict(
                    bin_names[f_idx],
                    ivs[pos : pos + f_bin_size],
                    else_ivs[f_idx],
                )
            )
            pos += f_bin_size
        return

    def _build_report(
        self,
        woes: Tuple[float],
        split_points: List[Union[np.ndarray, List[str]]],
        else_woes: List[Tuple],
        total_counts: List[int],
        else_counts: List[int],
    ) -> Dict:
        '''
        Attributes:
            woes: woe values for all features' bins.
            split_points: see _build_feature_bin.
            else_woe: woe values for all features' np.nan bin.
            total_counts: total samples all features' bins.
            else_counts: np.nan samples in all features.

        Return:
            Dict report
        '''
        assert len(else_woes) == len(self.bin_names), (
            f"len(else_woes) {len(else_woes)},"
            f" len(self.bin_names) {len(self.bin_names)}"
        )
        assert len(split_points) == len(self.bin_names), (
            f"len(split_points) {len(split_points)},"
            f" len(self.bin_names) {len(self.bin_names)}"
        )
        assert len(woes) == len(total_counts), (
            f"len(woes) {len(woes)}," f" len(total_counts) {len(total_counts)}"
        )
        assert len(else_woes) == len(else_counts), (
            f"len(else_woes) {len(else_woes)}," f" len(else_counts) {len(else_counts)}"
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

            assert pos + f_bin_size <= len(
                woes
            ), f"pos {pos}, f_bin_size {f_bin_size}, len(woes) {len(woes)}"
            variables.append(
                self._build_report_dict(
                    woes[pos : pos + f_bin_size],
                    self.bin_names[f_idx],
                    split_points[f_idx],
                    else_woes[f_idx],
                    total_counts[pos : pos + f_bin_size],
                    else_counts[f_idx],
                )
            )
            pos += f_bin_size

        assert pos == len(woes), f"pos {pos}, len(woes) {len(woes)}"
        assert len(variables) == len(self.bin_names), (
            f"len(variables) {len(variables)}, "
            f"len(self.bin_names) {len(self.bin_names)}"
        )
        return {"variables": variables}

    def label_holder_work(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        '''
        Label holder build report for it's own feature, and provide label to driver.
        Attributes:
            data: full dataset for this party.

        Return:
            Tuple[label, report for this party]
        '''
        bins_idx, split_points, else_bins = self._build_feature_bins(data)
        label = self._get_label(data)
        self.total_labels = label.size
        self.total_positives = round(label.sum())
        assert (
            self.total_positives != 0 and self.total_positives != self.total_labels
        ), "All label values are the same event"

        def sum_bin(bin):
            total_count = bin.size
            positive_count = round(label[bin].sum())
            return (total_count, positive_count)

        bins_stat = [sum_bin(b) for b in bins_idx]

        if self.binning_method == "chimerge":
            bins_stat, merged_split_point_indices = apply_chimerge(
                bins_stat,
                self.is_string_features(split_points),
                self.get_split_points_sizes(split_points),
                self.chimerge_target_bins,
                self.chimerge_target_chi,
            )
            split_points = update_split_points(
                split_points,
                merged_split_point_indices,
                self.is_string_features(split_points),
            )
        if len(self.bin_names) > 0:
            woes, bin_ivs = tuple(zip(*[self._calc_bin_woe_iv(*b) for b in bins_stat]))
            else_woes, else_ivs = tuple(
                zip(*[self._calc_bin_woe_iv(*sum_bin(b)) for b in else_bins])
            )
        else:
            woes, bin_ivs = [], []
            else_woes, else_ivs = [], []
        total_counts = [b[0] for b in bins_stat]
        else_counts = [b.size for b in else_bins]

        self.accumulate_iv_info(
            bin_ivs, self.bin_names, else_ivs, self.get_split_points_sizes(split_points)
        )
        return (
            label,
            self._build_report(
                woes, split_points, else_woes, total_counts, else_counts
            ),
        )

    def participant_build_sum_indices(self, data: pd.DataFrame) -> List[List[int]]:
        '''
        build sum indices for driver to calculate positive samples by HE.
        Attributes:
            data: full dataset for this party.

        Return:
            bin indices.
        '''
        bins_idx, self.split_points, else_bins_idx = self._build_feature_bins(data)
        self.total_counts = [b.size for b in bins_idx]
        self.else_counts = [b.size for b in else_bins_idx]
        return [*bins_idx, *[e for e in else_bins_idx if e.size]]

    def participant_build_sum_select(self, data: pd.DataFrame) -> np.ndarray:
        '''
        build select matrix for driver to calculate positive samples by Secret Sharing.
        Attributes:
            data: full dataset for this party.

        Return:
            sparse select matrix.
        '''
        bins_idx, self.split_points, else_bins_idx = self._build_feature_bins(data)
        self.total_counts = [b.size for b in bins_idx]
        self.else_counts = [b.size for b in else_bins_idx]

        samples = data.shape[0]
        select = np.zeros((samples, len(bins_idx)), np.float32)
        for i in range(len(bins_idx)):
            select[bins_idx[i], i] = 1.0

        else_select = list()
        for i in range(len(else_bins_idx)):
            if else_bins_idx[i].size:
                s = np.zeros((samples, 1), np.float32)
                s[else_bins_idx[i]] = 1.0
                else_select.append(s)

        return np.concatenate((select, *else_select), axis=1)

    def _is_string_features(self):
        return self.is_string_features(self.split_points)

    def is_string_features(self, split_points):
        return [isinstance(sp, list) for sp in split_points]

    def _get_split_points_sizes(self):
        return self.get_split_points_sizes(self.split_points)

    def get_split_points_sizes(self, split_points):
        return [len(sp) if isinstance(sp, list) else sp.size + 1 for sp in split_points]

    def get_bin_sum_info(self) -> 'ParticipantTransactionInfo':
        return ParticipantTransactionInfo(
            self.else_counts,
            self.total_counts,
            self.bin_names,
            self.binning_method,
            self._is_string_features(),
            self._get_split_points_sizes(),
            self.chimerge_target_bins,
            self.chimerge_target_chi,
        )

    def label_holder_sum_bin(
        self,
        bins_positive: Union[List, np.ndarray],
        bin_sum_info: 'ParticipantTransactionInfo',
    ) -> Tuple[List[Tuple[int, int]], List[int], List[Union[None, int]]]:
        """build bins stat tuple and information for participant to update

        Args:
            bins_positive (Union[List, np.ndarray]): number of positive samples in bins
            bin_sum_info ParticipantTransactionInfo: information participant give to label_holder.
        Returns:
            List[Tuple[int, int]: bin stats
            List[int]: updated total counts
            List[Union[None, int]]]: merged indices for split points
        """
        else_bin_count = len([x for x in bin_sum_info.else_counts if x > 0])
        if len(bins_positive) == 1 and isinstance(bins_positive[0], np.ndarray):
            bins_positive = list(bins_positive[0])
        else:
            bins_positive = list(bins_positive)

        if else_bin_count:
            else_positive = bins_positive[-else_bin_count:]
            bins_positive = bins_positive[:-else_bin_count]
        else:
            else_positive = list()

        assert len(bins_positive) == len(bin_sum_info.total_counts), (
            f"len(bins_positive) {len(bins_positive)}, "
            f"len(total_counts) {len(bin_sum_info.total_counts)}"
        )
        bins_positive = [round(float(p)) for p in bins_positive]
        bins_stat = [b for b in zip(bin_sum_info.total_counts, bins_positive)]
        total_counts = bin_sum_info.total_counts
        merged_split_point_indices = None
        if bin_sum_info.binning_method == "chimerge":
            bins_stat, merged_split_point_indices = apply_chimerge(
                bins_stat,
                bin_sum_info.is_string_features,
                bin_sum_info.split_points_sizes,
                bin_sum_info.chimerge_target_bins,
                bin_sum_info.chimerge_target_chi,
            )
            total_counts = [b[0] for b in bins_stat]

        else_stats = self._label_holder_build_else_stats(
            else_positive, bin_sum_info.bin_names, bin_sum_info.else_counts
        )
        bins_stat += else_stats

        return bins_stat, total_counts, merged_split_point_indices

    def participant_update_info(self, total_counts, merged_split_point_indices):
        if self.binning_method == "chimerge":
            self.total_counts = total_counts
            self.split_points = update_split_points(
                self.split_points,
                merged_split_point_indices,
                self._is_string_features(),
            )

    def _label_holder_build_else_stats(self, else_positive, bin_names, else_counts):
        assert len(bin_names) == len(else_counts), (
            f"len(bin_names) {len(bin_names)}, " f"len(else_counts) {len(else_counts)}"
        )

        else_stat = list()
        for i in range(len(else_counts)):
            count = else_counts[i]
            if count > 0:
                assert (
                    len(else_positive) > 0
                ), f"len(else_positive) {len(else_positive)}"
                p = else_positive.pop(0)
                else_stat.append((count, round(float(p))))
            else:
                else_stat.append((0, 0))
        assert len(else_positive) == 0, f"len(else_positive) {len(else_positive)}"

        return else_stat

    def label_holder_calc_woe_for_peer(
        self, bins_stat: List[Tuple[int, int]]
    ) -> Tuple[Tuple[float], Tuple[float]]:
        '''
        calculate woe/iv for participant party.
        Attributes:
            bins_stat: bins stat tuple from participant party.

        Return:
           woes : woe for each bin
           ivs : iv for each bin
        '''
        woe_iv = tuple(zip(*[self._calc_bin_woe_iv(*b) for b in bins_stat]))
        # empty case
        if len(woe_iv) != 2:
            return [], []
        woes, bin_ivs = woe_iv[0], woe_iv[1]
        return woes, bin_ivs

    def label_holder_collect_iv_for_participant(
        self, ivs: Tuple[float], transaction_info: 'ParticipantTransactionInfo'
    ):
        f_count = len(transaction_info.bin_names)
        self.accumulate_iv_info(
            ivs[:-f_count],
            transaction_info.bin_names,
            ivs[-f_count:],
            transaction_info.split_points_sizes,
        )

    def generate_iv_report(self, report_dict: Dict) -> Dict:
        iv_results = copy.deepcopy(self.iv_results)
        self.iv_results = []
        report_dict["feature_iv_info"] = iv_results
        return report_dict

    def participant_build_report(self, woes: Tuple[float]) -> Dict:
        '''
        build report based on label_holder party's woe values.
        Attributes:
            woes: woe values for all features' bins.

        Return:
            Dict
        '''
        f_count = len(self.bin_names)
        return self._build_report(
            woes[:-f_count],
            self.split_points,
            woes[-f_count:],
            self.total_counts,
            self.else_counts,
        )


@dataclass
class ParticipantTransactionInfo:
    """The information participant give to label_holder in order to calculate woe.

    All these information are public any ways or can be inferred from public information,
    except for total counts and else counts: these are revealed due to the approach of calculating woe,
    and yet we consider them ok to give to label_holder.

    Note that split point values are protected.


    else_counts (List[int]): number of nan values in bins
    total_counts (List[int]): total number of samples in bins
    bin_names (List[str]): bin names
    binning_method (str): binning method. 'chimerge' or else.
    is_string_features (List[bool]): if features are string type
    split_points_sizes (List[int]): size of split points
    chimerge_target_bins (int): chimerge parameter: target bin num
    chimerge_target_chi (float): chimerge parameter: chi
    """

    else_counts: List[int]
    total_counts: List[int]
    bin_names: List[str]
    binning_method: str
    is_string_features: List[bool]
    split_points_sizes: List[int]
    chimerge_target_bins: int
    chimerge_target_chi: float
