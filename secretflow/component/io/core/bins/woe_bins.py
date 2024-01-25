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
import logging
from typing import Dict, List

import numpy as np
from google.protobuf.json_format import MessageToJson, Parse

from secretflow.component.io.core.bins.bin_utils import (
    get_bin_label_holder_index,
    get_feature_iv_dict,
    get_index_feature_name_from_variable_list,
    get_party_features,
    pad_inf_to_split_points,
    strip_inf_from_split_points,
)
from secretflow.component.io.core.bins.woe_bin_utils import (
    calculate_woe_from_ratios,
    compute_bin_ratios,
    dispath_rules,
    merge_rules,
)
from secretflow.device import PYUObject
from secretflow.device.driver import reveal
from secretflow.spec.extend.bin_data_pb2 import Bin, Bins, VariableBins

MIN_SECURE_BIN_NUM = 5


def woe_feature_to_pb(
    feature: Dict, party_name: str, feature_iv: float
) -> VariableBins:
    assert feature["type"] == "numeric", "Only support numeric feature now."
    # Create a VariableBins message
    variable_bins = VariableBins()

    # Set the feature_name and feature_type
    variable_bins.feature_name = feature["name"]
    variable_bins.feature_type = feature["type"]

    bin_count = len(feature["filling_values"])
    if bin_count >= MIN_SECURE_BIN_NUM:
        logging.warning(
            f"DANGER! LABEL INFO IS AT RISK! For securing label information, bin count must be greater than or equal to {MIN_SECURE_BIN_NUM} to show, got {bin_count}."
        )
    assert bin_count > 1, "bin count should be at least 2"
    split_points_padded = pad_inf_to_split_points(feature["split_points"])
    assert (
        len(split_points_padded) == bin_count + 1
    ), f"bin count should be will defined, filling values: {feature['filling_values']}, {feature['split_points']}"
    for i in range(bin_count):
        # Create a Bin message
        bin = Bin()
        bin.left_bound = split_points_padded[i]
        bin.right_bound = split_points_padded[i + 1]
        bin.total_count = feature["total_counts"][i]
        bin.filling_value = feature["filling_values"][i]
        variable_bins.valid_bins.append(bin)

    else_bin = Bin()
    else_bin.left_bound = 0
    else_bin.right_bound = 0
    else_bin.total_count = feature["else_counts"]
    else_bin.filling_value = feature["else_filling_value"]
    variable_bins.else_bin.CopyFrom(else_bin)
    variable_bins.valid_bin_count = bin_count
    variable_bins.iv = feature_iv
    variable_bins.is_woe = True
    variable_bins.party_name = party_name
    return variable_bins


def woe_bin_rule_to_pb(rules: List[PYUObject], label_holder_index: int) -> Bins:
    """convert a woe bin rule into pb
    Sample WOE rule dict:
    {
        "variables":[
            {
                "name": str, # feature name
                "type": str, # "string" or "numeric", if feature is discrete or continuous
                "categories": list[str], # categories for discrete feature
                "split_points": list[float], # left-open right-close split points
                "total_counts": list[int], # total samples count in each bins.
                "else_counts": int, # np.nan samples count
                "filling_values": list[float], # woe values for each bins.
                "else_filling_value": float, # woe value for np.nan samples.
            },
            # ... others feature
        ],
        # label holder's PYUObject only
        # warning: giving bin_ivs to other party will leak positive samples in each bin.
        # it is up to label holder's will to give feature iv or bin ivs or all info to workers.
        # for more information, look at: https://github.com/secretflow/secretflow/issues/565

        # in the following comment, by safe we mean label distribution info is not leaked.
        "feature_iv_info" :[
            {
                "name": str, #feature name
                "ivs": list[float], #iv values for each bins, not safe to share with workers in any case.
                "else_iv": float, #iv for nan values, may share to with workers
                "feature_iv": float, #sum of bin_ivs, safe to share with workers when bin num > 2.
            }
        ]
    }
    """
    bins = Bins()
    feature_iv_dict = get_feature_iv_dict(rules[label_holder_index])
    for i, rule in enumerate(rules):
        # Iterate through each variable in the data
        party_name = rules[i].device.party
        variable_data_list = reveal(rule.device(lambda x: x["variables"])(rule))

        for variable_data in variable_data_list:
            if variable_data["type"] == "string":
                continue
            bins.variable_bins.append(
                woe_feature_to_pb(
                    variable_data, party_name, feature_iv_dict[variable_data["name"]]
                )
            )

    return bins


def feature_modify_woe_bin_rule(
    feature_rule: Dict, new_variable_rule_pb: VariableBins
) -> Dict:
    if feature_rule["type"] != "numeric":
        return feature_rule
    assert (
        feature_rule["name"] == new_variable_rule_pb.feature_name
    ), "feature name does not match"
    assert (
        feature_rule["type"] == new_variable_rule_pb.feature_type
    ), "feature type does not match"

    assert len(feature_rule["total_counts"]) == len(
        new_variable_rule_pb.valid_bins
    ), "bin number should match"

    bin_ratios = feature_rule["bin_ratios"]

    new_split_points = [new_variable_rule_pb.valid_bins[0].left_bound]
    new_total_counts = []
    new_bin_ratio_pairs = []

    have_previous_mark = False
    cached_total_counts = 0
    cached_bin_ratio_pair = np.array([0.0, 0.0])
    last_seen_right_bound = None
    # assumed bins are arranged in order with the last bin being nan bin.
    for i, bin in enumerate(new_variable_rule_pb.valid_bins):
        bin_ratio_pair = np.array(bin_ratios[i])
        if bin.mark_for_merge is True:
            # does not add right bound now, instead, check if can merge and update right bound cache.
            if not have_previous_mark:
                # indicate a new merging process begins
                cached_total_counts = 0
                cached_bin_ratio_pair = np.array([0.0, 0.0])
                last_seen_right_bound = None
            else:
                this_left_bound = bin.left_bound
                assert (
                    this_left_bound == last_seen_right_bound
                ), f"only consecutive bins can merge, last right bound :{last_seen_right_bound}, this bin's left bound : {this_left_bound}"
            # merging
            last_seen_right_bound = bin.right_bound
            cached_total_counts += bin.total_count
            cached_bin_ratio_pair += bin_ratio_pair
            have_previous_mark = True
        else:
            # this bin is not merging, right bounds should be added.
            if have_previous_mark:
                # indicate ending of the previous merging process
                new_split_points.append(last_seen_right_bound)
                new_total_counts.append(cached_total_counts)
                new_bin_ratio_pairs.append(cached_bin_ratio_pair)
            # no matter what is the previous mark, this bin's right bound is surely added.
            new_split_points.append(bin.right_bound)
            new_total_counts.append(bin.total_count)
            new_bin_ratio_pairs.append(bin_ratio_pair)
            have_previous_mark = False

    # consider a bin for nan values
    if have_previous_mark:
        # indicate ending of the previous merging process
        new_split_points.append(last_seen_right_bound)
        new_total_counts.append(cached_total_counts)
        new_bin_ratio_pairs.append(cached_bin_ratio_pair)
    # check bin size >= MIN_SECURE_BIN_NUM
    bin_count = len(new_bin_ratio_pairs)
    if bin_count >= MIN_SECURE_BIN_NUM:
        logging.warning(
            f"DANGER! LABEL INFO IS AT RISK! For securing label information, bin count must be greater than or equal to {MIN_SECURE_BIN_NUM} to show, got {bin_count}."
        )
    assert bin_count > 1, "bin count should be at least 2"

    feature_rule["filling_values"] = [
        calculate_woe_from_ratios(rp, rn) for (rp, rn) in new_bin_ratio_pairs
    ]
    feature_rule["split_points"] = strip_inf_from_split_points(new_split_points)
    feature_rule["total_counts"] = new_total_counts
    return feature_rule


def party_modify_woe_bin_rule(rule: Dict, bins_json: str) -> Dict:
    """Suppose only label holder party execute this.

    rule is a merged rule with computed bin_ratios.
    """
    assert "variables" in rule, "rules must contain variables"
    bins = Bins()
    Parse(bins_json, bins)
    party_variable_bins: List[VariableBins] = bins.variable_bins
    new_variables = []
    # assume the order of features should be left unchanged
    for variable_data in rule["variables"]:
        if variable_data["type"] == "string":
            new_variables.append(variable_data)
        else:
            new_variable_rule_pb = party_variable_bins[
                get_index_feature_name_from_variable_list(
                    party_variable_bins, variable_data["name"]
                )
            ]
            new_variables.append(
                feature_modify_woe_bin_rule(variable_data, new_variable_rule_pb)
            )
    rule["variables"] = new_variables
    return rule


def woe_bin_rule_from_pb_and_old_rule(
    rules: List[PYUObject], bins: Bins
) -> List[PYUObject]:
    """construct a woe bin rule from pb and old rule"""
    # the idea is
    # 1. record variable belongings.
    # 2. send all rules to label holder party, which will compute everything
    #     1. recompute pos bins beforehand.
    #     2. do aggregations with pos bins
    # 3. sends back the rules belonging to each party.

    # step 1
    party_feature_map = get_party_features(rules)
    label_holder_index: int = get_bin_label_holder_index(rules)
    label_holder_device = rules[label_holder_index].device
    party_order = [rule.device.party for rule in rules]

    # step 2
    rules_to_label = [rule.to(label_holder_device) for rule in rules]
    merged_rule_object = label_holder_device(merge_rules)(rules_to_label)

    # step 2.1
    merged_rule_with_bin_ratios_object = label_holder_device(compute_bin_ratios)(
        merged_rule_object
    )

    # step 2.2
    # note that bins pb must be serialized to json before passing into ray
    updated_merged_rule_object = label_holder_device(party_modify_woe_bin_rule)(
        merged_rule_with_bin_ratios_object, MessageToJson(bins, indent=0)
    )

    # step 3
    label_rule, all_rules = label_holder_device(dispath_rules, num_returns=2)(
        updated_merged_rule_object, party_feature_map, label_holder_index, party_order
    )
    new_rules = []
    for i, rule in enumerate(rules):
        if i == label_holder_index:
            new_rules.append(label_rule)
        else:
            new_rules.append(
                label_holder_device(lambda x: x[i])(all_rules).to(rule.device)
            )
    return new_rules
