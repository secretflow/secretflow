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
from typing import Dict, List

from google.protobuf.json_format import MessageToJson, Parse

from secretflow.component.io.core.bins.bin_utils import (
    pad_inf_to_split_points,
    strip_inf_from_split_points,
)
from secretflow.device import PYUObject
from secretflow.device.driver import reveal
from secretflow.spec.extend.bin_data_pb2 import Bin, Bins, VariableBins


def normal_feature_to_pb(feature: Dict, party_name: str) -> VariableBins:
    assert feature["type"] == "numeric", "Only support numeric feature now."
    # Create a VariableBins message
    variable_bins = VariableBins()

    # Set the feature_name and feature_type
    variable_bins.feature_name = feature["name"]
    variable_bins.feature_type = feature["type"]

    bin_count = len(feature["filling_values"])
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
    variable_bins.is_woe = False
    variable_bins.party_name = party_name
    return variable_bins


def normal_bin_rule_to_pb(rules: List[PYUObject]) -> Bins:
    """convert a normal bin rule into pb"""
    bins = Bins()

    rules_data = reveal(rules)

    for i, data in enumerate(rules_data):
        # Iterate through each variable in the data
        party_name = rules[i].device.party
        variable_data_list = data["variables"] if "variables" in data else []
        for variable_data in variable_data_list:
            if variable_data["type"] == "string":
                continue
            bins.variable_bins.append(normal_feature_to_pb(variable_data, party_name))

    return bins


def feature_modify_normal_bin_rule(
    feature_rule: Dict, new_variable_rule_pb_json: str
) -> Dict:
    # this is because ray cannot serialize pb message, we packed it into json and parse it back to pb here
    new_variable_rule_pb = VariableBins()
    Parse(new_variable_rule_pb_json, new_variable_rule_pb)
    if feature_rule["type"] != "numeric":
        return feature_rule
    assert (
        feature_rule["name"] == new_variable_rule_pb.feature_name
    ), "feature name does not match"
    assert (
        feature_rule["type"] == new_variable_rule_pb.feature_type
    ), "feature type does not match"

    new_split_points = [new_variable_rule_pb.valid_bins[0].left_bound]
    new_total_counts = []

    have_previous_mark = False
    cached_total_counts = 0
    last_seen_right_bound = None
    # assumed bins are arranged in order.
    for bin in new_variable_rule_pb.valid_bins:
        if bin.mark_for_merge is True:
            # does not add right bound now, instead, check if can merge and update right bound cache.
            if not have_previous_mark:
                # indicate a new merging process begins
                cached_total_counts = 0
                last_seen_right_bound = None
            else:
                this_left_bound = bin.left_bound
                assert (
                    this_left_bound == last_seen_right_bound
                ), f"only consecutive bins can merge, last right bound :{last_seen_right_bound}, this bin's left bound : {this_left_bound}"
            # merging
            last_seen_right_bound = bin.right_bound
            cached_total_counts += bin.total_count
            have_previous_mark = True
        else:
            # this bin is not merging, right bounds should be added.
            if have_previous_mark:
                # indicate ending of the previous merging process
                new_split_points.append(last_seen_right_bound)
                new_total_counts.append(cached_total_counts)
            # no matter what is the previous mark, this bin's right bound is surely added.
            new_split_points.append(bin.right_bound)
            new_total_counts.append(bin.total_count)
            have_previous_mark = False

    if have_previous_mark:
        new_split_points.append(last_seen_right_bound)
        new_total_counts.append(cached_total_counts)
    # check bin size > 2
    assert len(new_split_points) > 2, "bin size should be greater than 2"
    feature_rule["split_points"] = strip_inf_from_split_points(new_split_points)
    feature_rule["total_counts"] = new_total_counts
    feature_rule["filling_values"] = [*range(len(new_total_counts))]
    return feature_rule


def party_modify_normal_bin_rule(
    rule: Dict, party_variable_bins: List[VariableBins]
) -> Dict:
    """Assume rules is of form:
    {
        "variables":[
            {
                "name": str, # feature name
                "type": str, # "string" or "numeric", if feature is discrete or continuous
                "categories": list[str], # categories for discrete feature
                "split_points": list[float], # left-open right-close split points
                "total_counts": list[int], # total samples count in each bins.
                "else_counts": int, # np.nan samples count
                # for this binning method, we use [*range(f_num_bins)] as filling values
                # that is 0 for bin with index 0, 1 for bin with index 1 etc.
                "filling_values": list[float], # filling values for each bins.
                # for this binning method, we use -1 as filling value for nan samples
                "else_filling_value": float, # filling value for np.nan samples.
            },
            # ... others feature
        ],
    }
    """
    assert "variables" in rule, "rules must contain variables"
    new_variables = []
    i = 0
    # assume the order of features should be left unchanged
    for variable_data in rule["variables"]:
        if variable_data["type"] == "string":
            new_variables.append(variable_data)
        else:
            new_variable_rule_pb = party_variable_bins[i]
            new_variables.append(
                feature_modify_normal_bin_rule(variable_data, new_variable_rule_pb)
            )
            i += 1
    rule["variables"] = new_variables
    return rule


def normal_bin_rule_from_pb_and_old_rule(
    rules: List[PYUObject], bins: Bins
) -> List[PYUObject]:
    """construct a normal bin rule from pb and old rule"""
    variable_bins = bins.variable_bins
    new_rules = []
    for i, rule in enumerate(rules):
        # Iterate through each variable in the data
        party_name = rules[i].device.party
        party_variable_list = [
            MessageToJson(variable_bin, indent=0)
            for variable_bin in variable_bins
            if variable_bin.party_name == party_name
        ]
        # ray serialization cannot deal with pb objects, parse it into dict before continue
        new_rules.append(
            rule.device(party_modify_normal_bin_rule)(rule, party_variable_list)
        )
    return new_rules
