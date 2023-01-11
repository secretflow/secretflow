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

from typing import List, Dict, Union, Tuple
from scipy.stats import chi2
import pandas as pd
import numpy as np
import math

from secretflow.device import PYUObject, proxy


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
                if self.binning_method == "quantile"
                else self.chimerge_init_bins
            )
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

    def _build_report_dict(
        self,
        woe_ivs: List[Tuple],
        f_name: str,
        split_points: Union[np.ndarray, List[str]],
        else_woe_iv: Tuple,
        total_counts: List[int],
        else_counts: int,
    ) -> Dict:
        '''
        build report dict for one feature.
        Attributes:
            woe_ivs: woe/iv values for each bins in feature.
            f_name: feature name.
            split_points: see _build_feature_bin.
            else_woe_iv: woe/iv for np.nan values in feature.
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

        ret['woes'] = list()
        ret['ivs'] = list()
        ret['total_counts'] = list()
        assert len(total_counts) == len(woe_ivs), (
            f"len(total_counts) {len(total_counts)}," f" len(woe_ivs) {len(woe_ivs)}"
        )
        for i in range(len(woe_ivs)):
            ret['total_counts'].append(total_counts[i])
            ret['woes'].append(woe_ivs[i][0])
            ret['ivs'].append(woe_ivs[i][1])

        ret['else_woe'] = else_woe_iv[0]
        ret['else_iv'] = else_woe_iv[1]
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

    def _build_report(
        self,
        woe_ivs: List[Tuple[float]],
        split_points: List[Union[np.ndarray, List[str]]],
        else_woe_ivs: List[Tuple],
        total_counts: List[int],
        else_counts: List[int],
    ) -> Dict:
        '''
        Attributes:
            woe_ivs: woe/iv values for all features' bins.
            split_points: see _build_feature_bin.
            else_woe_iv: woe/iv values for all features' np.nan bin.
            total_counts: total samples all features' bins.
            else_counts: np.nan samples in all features.

        Return:
            Dict report
        '''
        assert len(else_woe_ivs) == len(self.bin_names), (
            f"len(else_woe_ivs) {len(else_woe_ivs)},"
            f" len(self.bin_names) {len(self.bin_names)}"
        )
        assert len(split_points) == len(self.bin_names), (
            f"len(split_points) {len(split_points)},"
            f" len(self.bin_names) {len(self.bin_names)}"
        )
        assert len(woe_ivs) == len(total_counts), (
            f"len(woe_ivs) {len(woe_ivs)}," f" len(total_counts) {len(total_counts)}"
        )
        assert len(else_woe_ivs) == len(else_counts), (
            f"len(else_woe_ivs) {len(else_woe_ivs)},"
            f" len(else_counts) {len(else_counts)}"
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
                woe_ivs
            ), f"pos {pos}, f_bin_size {f_bin_size}, len(woe_ivs) {len(woe_ivs)}"
            variables.append(
                self._build_report_dict(
                    woe_ivs[pos : pos + f_bin_size],
                    self.bin_names[f_idx],
                    split_points[f_idx],
                    else_woe_ivs[f_idx],
                    total_counts[pos : pos + f_bin_size],
                    else_counts[f_idx],
                )
            )
            pos += f_bin_size

        assert pos == len(woe_ivs), f"pos {pos}, len(woe_ivs) {len(woe_ivs)}"
        assert len(variables) == len(self.bin_names), (
            f"len(variables) {len(variables)}, "
            f"len(self.bin_names) {len(self.bin_names)}"
        )
        return {"variables": variables}

    def _chi_merge(
        self, bins: List[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], List[int]]:
        '''
        apply ChiMerge on one feature. ChiMerge proposed by paper AAAI92-019.
        merge adjacent bins by their samples' Chi-Square Statistic.
        Attributes:
            bins: bins in feature build by initialization cut.

        Return:
            Tuple[bins after merge, removed bin indices in input bins]
        '''

        def get_chi(bin1: Tuple[float, float], bin2: Tuple[float, float]):
            total = bin1[0] + bin2[0]
            total_positive = bin1[1] + bin2[1]
            positive_rate = float(total_positive) / float(total)

            if positive_rate == 0 or positive_rate == 1:
                # two bins has same label distribution
                return 0.0

            bin1_positive = bin1[1]
            bin1_expt_positive = positive_rate * float(bin1[0])
            bin1_negative = bin1[0] - bin1[1]
            bin1_expt_negative = float(bin1[0]) - bin1_expt_positive

            bin2_positive = bin2[1]
            bin2_expt_positive = positive_rate * float(bin2[0])
            bin2_negative = bin2[0] - bin2[1]
            bin2_expt_negative = float(bin2[0]) - bin2_expt_positive

            return (
                math.pow(bin1_positive - bin1_expt_positive, 2) / bin1_expt_positive
                + math.pow(bin1_negative - bin1_expt_negative, 2) / bin1_expt_negative
                + math.pow(bin2_positive - bin2_expt_positive, 2) / bin2_expt_positive
                + math.pow(bin2_negative - bin2_expt_negative, 2) / bin2_expt_negative
            )

        chis = [get_chi(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

        def get_min(chis):
            min_idx = np.argmin(chis)
            return chis[min_idx], min_idx

        orig_idx = [i for i in range(len(bins))]
        removed_idx = list()
        while True:
            if len(bins) <= self.chimerge_target_bins:
                # chi_merge stop by bin size
                return bins, removed_idx

            min_chi, min_idx = get_min(chis)
            if min_chi > self.chimerge_target_chi:
                # chi_merge stop by chi value3333333
                return bins, removed_idx

            # merge bins[min_idx] & bins[min_idx + 1]
            new_stat = (
                bins[min_idx][0] + bins[min_idx + 1][0],
                bins[min_idx][1] + bins[min_idx + 1][1],
            )
            bins.pop(min_idx + 1)
            bins[min_idx] = new_stat
            removed_idx.append(orig_idx.pop(min_idx))
            # update chis
            chis.pop(min_idx)
            if min_idx > 0:
                chis[min_idx - 1] = get_chi(bins[min_idx - 1], bins[min_idx])
            if min_idx < len(bins) - 1:
                chis[min_idx] = get_chi(bins[min_idx], bins[min_idx + 1])

    def _apply_chimerge(
        self,
        bins_stat: List[Tuple[float, float]],
        split_points: List[Union[np.ndarray, List[str]]],
    ) -> Tuple[List[Tuple[float, float]], List[Union[np.ndarray, List[str]]]]:
        '''
        apply ChiMerge on all number type features.
        Attributes:
            bins_stat: bins for all features build by initialization cut.
            split_points: see _build_feature_bin

        Return:
            Tuple[bins after merge, split points after merge]
        '''
        pos = 0
        merged_bins_stat = list()
        merged_split_points = list()
        for f_idx in range(len(split_points)):
            if isinstance(split_points[f_idx], list):
                # can not apple chimerge on string type feature. skip and forward values into result.
                f_size = len(split_points[f_idx])
                merged_bins_stat += bins_stat[pos : pos + f_size]
                merged_split_points.append(split_points[f_idx])
                pos += f_size
            else:
                f_size = split_points[f_idx].size + 1
                mbs, merged_idx = self._chi_merge(bins_stat[pos : pos + f_size])
                merged_split_points.append(np.delete(split_points[f_idx], merged_idx))
                merged_bins_stat += mbs
                pos += f_size

        return merged_bins_stat, merged_split_points

    def coordinator_work(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
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
            bins_stat, split_points = self._apply_chimerge(bins_stat, split_points)

        woe_ivs = [self._calc_bin_woe_iv(*s) for s in bins_stat]
        total_counts = [b[0] for b in bins_stat]

        else_woe_ivs = [self._calc_bin_woe_iv(*sum_bin(b)) for b in else_bins]
        else_counts = [b.size for b in else_bins]

        return (
            label,
            self._build_report(
                woe_ivs, split_points, else_woe_ivs, total_counts, else_counts
            ),
        )

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

    def participant_sum_bin(
        self, bins_positive: Union[List, np.ndarray]
    ) -> List[Tuple[int, int]]:
        '''
        build bins stat tuple.
        Attributes:
            bins_positive: positive counts in all not empty bins.

        Return:
            List[Tuple[total, positive]]
        '''
        else_bin_count = len([x for x in self.else_counts if x > 0])
        if len(bins_positive) == 1 and isinstance(bins_positive[0], np.ndarray):
            bins_positive = list(bins_positive[0])
        else:
            bins_positive = list(bins_positive)

        if else_bin_count:
            else_positive = bins_positive[-else_bin_count:]
            bins_positive = bins_positive[:-else_bin_count]
        else:
            else_positive = list()

        assert len(bins_positive) == len(self.total_counts), (
            f"len(bins_positive) {len(bins_positive)}, "
            f"len(self.total_counts) {len(self.total_counts)}"
        )

        bins_positive = [round(float(p)) for p in bins_positive]
        bins_stat = [b for b in zip(self.total_counts, bins_positive)]

        if self.binning_method == "chimerge":
            bins_stat, self.split_points = self._apply_chimerge(
                bins_stat, self.split_points
            )
            self.total_counts = [b[0] for b in bins_stat]

        assert len(self.bin_names) == len(self.else_counts), (
            f"len(self.bin_names) {len(self.bin_names)}, "
            f"len(self.else_counts) {len(self.else_counts)}"
        )

        else_stat = list()
        for i in range(len(self.else_counts)):
            count = self.else_counts[i]
            if count > 0:
                assert (
                    len(else_positive) > 0
                ), f"len(else_positive) {len(else_positive)}"
                p = else_positive.pop(0)
                else_stat.append((count, round(float(p))))
            else:
                else_stat.append((0, 0))
        assert len(else_positive) == 0, f"len(else_positive) {len(else_positive)}"

        bins_stat += else_stat

        return bins_stat

    def coordinator_calc_woe_for_peer(
        self, bins_stat: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        '''
        calculate woe/iv for participant party.
        Attributes:
            bins_stat: bins stat tuple from participant party.

        Return:
           List[Tuple[woe, iv]]
        '''
        return [self._calc_bin_woe_iv(*b) for b in bins_stat]

    def participant_build_report(self, woe_ivs: List[Tuple[float, float]]) -> Dict:
        '''
        build report based on coordinator party's woe/iv values.
        Attributes:
            woe_ivs: woe/iv values for all features' bins.

        Return:
            Dict
        '''
        f_count = len(self.bin_names)
        return self._build_report(
            woe_ivs[:-f_count],
            self.split_points,
            woe_ivs[-f_count:],
            self.total_counts,
            self.else_counts,
        )
