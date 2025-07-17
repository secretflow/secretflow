# Copyright 2025 Ant Group Co., Ltd.
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

from typing import Dict, List, Tuple


def agg_gain_statistics(gain_stats_list: List[Tuple[Dict, Dict]]) -> Dict:
    """Aggregate gain statistics from different trees

    The idea is the same as XGBoost Gain feature importance.
    However, our gain is calculated differently, so the numeric values are different.

    Gain is the improvement in accuracy brought by a feature to the branches it is on.
    The idea is that before adding a new split on a feature X to the branch there were some wrongly classified elements;
    after adding the split on this feature, there are two new branches,
    and each of these branches is more accurate
    (one branch saying if your observation is on this branch then it should be classified as 1,
    and the other branch saying the exact opposite).

    """
    gain_sums = {}
    gain_count = {}
    for tree_gain_sum, tree_gain_count in gain_stats_list:
        for feature_name, gain in tree_gain_sum.items():
            if feature_name == -1:
                continue
            if feature_name not in gain_sums:
                gain_sums[feature_name] = 0
                gain_count[feature_name] = 0
            gain_sums[feature_name] += gain
            gain_count[feature_name] += tree_gain_count[feature_name]

    return {k: v / max([gain_count[k], 1]) for k, v in gain_sums.items()}


def agg_split_statistics(gain_stats_list: List[Tuple[Dict, Dict]]) -> Dict:
    """Aggregate split count statistics from different trees.

    The idea is the same as XGBoost 'Frequency' feature importance. the other name for it is 'weight'.

    Frequency is a simpler way to measure the Gain.
    It just counts the number of times a feature is used in all generated trees.
    You should not use it (unless you know why you want to use it).

    """
    gain_count = {}
    for tree_gain_sum, tree_gain_count in gain_stats_list:
        for feature_name, _ in tree_gain_sum.items():
            if feature_name == -1:
                continue
            if feature_name not in gain_count:
                gain_count[feature_name] = 0
            gain_count[feature_name] += tree_gain_count[feature_name]

    return gain_count


SUPPORTED_IMPORTANCE_TYPE_STATS = {
    "gain": agg_gain_statistics,
    "weight": agg_split_statistics,
}

SUPPORTED_IMPORTANCE_DESCRIPTIONS = {
    'gain': 'the average gain across all splits the feature is used in.',
    'weight': 'the number of times a feature is used to split the data across all trees.',
}
