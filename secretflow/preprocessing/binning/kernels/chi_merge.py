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


def chi_merge(
    bins: List[Tuple[float, float]],
    chimerge_target_bins: int,
    chimerge_target_chi: float,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    '''
    apply ChiMerge on one feature. ChiMerge proposed by paper AAAI92-019.
    merge adjacent bins by their samples' Chi-Square Statistic.
    Attributes:
        bins: bins in feature build by initialization cut.
        chimerge_target_bins: target number of bins
        chimerge_target_chi: target chi

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
        if len(bins) <= chimerge_target_bins:
            # chi_merge stop by bin size
            return bins, removed_idx

        min_chi, min_idx = get_min(chis)
        if min_chi > chimerge_target_chi:
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


def apply_chimerge(
    bins_stat: List[Tuple[float, float]],
    is_string_features: List[bool],
    split_points_sizes: List[int],
    chimerge_target_bins: int,
    chimerge_target_chi: float,
) -> Tuple[List[Tuple[float, float]], List[Union[None, int]]]:
    '''
    apply ChiMerge on all number type features.
    Attributes:
        bins_stat: bins for all features build by initialization cut.
        is_string_features: if feature is string type.
        split_points_sizes: size of split points
        chimerge_target_bins: target number of bins
        chimerge_target_chi: target chi
    Return:
        Tuple[bins after merge, split point indices for merge]
    '''
    pos = 0
    merged_bins_stat = []
    merged_split_point_indices = []
    for is_string_type, f_size in zip(is_string_features, split_points_sizes):
        if is_string_type:
            # can not apple chimerge on string type feature. skip and forward values into result.
            merged_bins_stat += bins_stat[pos : pos + f_size]
            pos += f_size
            merged_split_point_indices.append(None)
        else:
            mbs, merged_idx = chi_merge(
                bins_stat[pos : pos + f_size], chimerge_target_bins, chimerge_target_chi
            )
            merged_split_point_indices.append(merged_idx)
            merged_bins_stat += mbs
            pos += f_size

    return merged_bins_stat, merged_split_point_indices


def update_split_points(split_points, merged_split_point_indices, is_string_features):
    merged_split_points = []
    for is_string_type, sp, merged_idx in zip(
        is_string_features, split_points, merged_split_point_indices
    ):
        if is_string_type:
            merged_split_points.append(sp)
        else:
            merged_split_points.append(np.delete(sp, merged_idx))
    return merged_split_points
