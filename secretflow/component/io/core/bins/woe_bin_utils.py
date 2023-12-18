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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.∏
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

import numpy as np


def merge_rules(rules: List[Dict]) -> Dict:
    """Merge multiple rule dict into one"""
    new_dict = {}
    new_variables = []
    for rule in rules:
        new_variables.extend(rule["variables"])
        if "feature_iv_info" in rule:
            new_dict["feature_iv_info"] = rule["feature_iv_info"]

    new_dict["variables"] = new_variables
    return new_dict


def find_feature_index(variables: List[Dict], feature_name):
    for i, variable in enumerate(variables):
        if variable['name'] == feature_name:
            return i
    raise ValueError("Feature name not found")


def dispath_rules(
    rule: Dict,
    party_features: Dict[str, List[str]],
    label_holder_index: int,
    party_order: List[str],
) -> Tuple[Dict, List[Dict]]:
    """Dispatch rules to different parties"""
    result_list = []
    label_dict = {}
    variables = rule["variables"]
    for i, party in enumerate(party_order):
        new_rule_dict = {}
        new_variables = []
        for feature in party_features[party]:
            feature_index = find_feature_index(variables, feature)
            new_variables.append(variables.pop(feature_index))
        new_rule_dict["variables"] = new_variables
        if i == label_holder_index:
            new_rule_dict["feature_iv_info"] = rule["feature_iv_info"]
            label_dict = new_rule_dict
            result_list.append({})
        else:
            result_list.append(new_rule_dict)
    return label_dict, result_list


def calculate_woe_from_ratios(rp: float, rn: float):
    return np.log(rp / rn)


FLOAT_TOLERANCE = 1e-6


def calculate_bin_ratios(
    bin_iv: float, bin_woe: float, bin_total: int, feature_total: int
) -> Tuple[float, float]:
    """
    Calculate two ratios: rp = bin_pos/total_positive, rn = bin_negative/total negative.

    because new_woe = log(new_rp / new_rn).

    Recall:
        bin_count = bin_positives + bin_negatives
        total_count = total_positives + total_negatives
        bin_woe = log((bin_positives / total_positives) / (bin_negatives / total_negatives))
        bin_iv = ((bin_positives / total_positives) - (bin_negatives / total_negatives)) * bin_woe
    """
    if abs(bin_woe) <= FLOAT_TOLERANCE:
        x = bin_total * 1.0 / feature_total
        y = x
        return x, y
    D = bin_iv / bin_woe
    y = D / (np.exp(bin_woe) - 1)
    x = y + D
    return x, y


def compute_bin_ratios(merged_rule: Dict) -> Dict:
    """Based on the merged rule and bin ivs, compute all bin ratios"""
    original_variables: List[Dict] = merged_rule["variables"]
    feature_iv_info: List[Dict] = merged_rule["feature_iv_info"]
    new_variables: List[Dict] = []
    for feature_iv in feature_iv_info:
        feature_name = feature_iv["name"]
        feature_index = find_feature_index(original_variables, feature_name)
        variable = original_variables.pop(feature_index)
        bin_counts = variable["total_counts"]
        woes = variable["filling_values"]
        ivs = feature_iv["ivs"]
        num_bins = len(bin_counts)
        feature_total = sum(bin_counts)
        bin_ratios = [
            calculate_bin_ratios(
                bin_woe=woes[i],
                bin_iv=ivs[i],
                bin_total=bin_counts[i],
                feature_total=feature_total,
            )
            for i in range(num_bins)
        ]
        variable["bin_ratios"] = bin_ratios
        new_variables.append(variable)
    merged_rule["variables"] = new_variables
    return merged_rule
