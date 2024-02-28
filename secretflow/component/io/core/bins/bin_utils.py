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

import numpy as np

from secretflow.device import PYUObject
from secretflow.device.driver import reveal
from secretflow.spec.extend.bin_data_pb2 import VariableBins


def check_rule_dict_is_from_woe_label_holder(rule: PYUObject) -> bool:
    return reveal(
        rule.device(lambda rule_dict: "feature_iv_info" in rule_dict.keys())(rule)
    )


def get_bin_label_holder_index(rules: List[PYUObject]) -> int:
    for i, rule in enumerate(rules):
        if check_rule_dict_is_from_woe_label_holder(rule):
            return i
    return -1


def check_bin_rule_is_woe(index: int) -> bool:
    """check if a bin rule model is woe rule or normal bin rule"""
    if index != -1:
        return True
    else:
        return False


def feature_iv_list_to_dict(feature_iv_list: List[Dict]) -> Dict:
    """convert feature iv list to dict"""
    feature_iv_dict = {}
    for feature_iv in feature_iv_list:
        feature_iv_dict[feature_iv["name"]] = feature_iv["feature_iv"]
    return feature_iv_dict


def get_feature_iv_dict(label_holder_rule: PYUObject) -> Dict:
    """get feature iv dict from label holder rule"""
    feature_iv_dict = reveal(
        label_holder_rule.device(
            lambda rule_dict: feature_iv_list_to_dict(rule_dict["feature_iv_info"])
        )(label_holder_rule)
    )
    return feature_iv_dict


def get_index_feature_name_from_variable_list(
    variable_list: List[VariableBins], feature_name: str
):
    for i, variable in enumerate(variable_list):
        if variable.feature_name == feature_name:
            return i
    raise ValueError("no such feature name")


def get_party_features(rules: List[PYUObject]) -> Dict[str, List[str]]:
    """get party features from rules"""
    party_features = {}
    for rule in rules:
        feature_names = reveal(
            rule.device(lambda r: [variable["name"] for variable in r["variables"]])(
                rule
            )
        )
        party_features[rule.device.party] = feature_names
    return party_features


def pad_inf_to_split_points(split_points: List[float]) -> List[float]:
    assert isinstance(split_points, list), f"{split_points}"
    return [-np.inf] + split_points + [np.inf]


def strip_inf_from_split_points(split_points: List[float]) -> List[float]:
    assert (
        split_points[0] == -np.inf and split_points[-1] == np.inf
    ), f"split points head, end and all :{split_points[0], split_points[1], split_points}"
    return split_points[1:-1]
