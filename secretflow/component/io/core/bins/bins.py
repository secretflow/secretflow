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
from typing import List

from secretflow.component.io.core.bins.bin_utils import (
    check_bin_rule_is_woe,
    get_bin_label_holder_index,
)
from secretflow.component.io.core.bins.vert_bins import (
    normal_bin_rule_from_pb_and_old_rule,
    normal_bin_rule_to_pb,
)
from secretflow.component.io.core.bins.woe_bins import (
    woe_bin_rule_from_pb_and_old_rule,
    woe_bin_rule_to_pb,
)
from secretflow.device import PYUObject
from secretflow.spec.extend.bin_data_pb2 import Bins


def bin_rule_to_pb(rules: List[PYUObject], public_info: dict) -> Bins:
    """Convert part of the public info in rules to Bins pb"""

    label_holder_index = get_bin_label_holder_index(rules)
    bin_pb = (
        woe_bin_rule_to_pb(rules, label_holder_index)
        if check_bin_rule_is_woe(label_holder_index)
        else normal_bin_rule_to_pb(rules)
    )
    bin_pb.model_hash = public_info.get("model_hash", "")
    return bin_pb


def bin_rule_from_pb_and_old_rule(
    rules: List[PYUObject], public_info: dict, bins: Bins
) -> List[PYUObject]:
    label_holder_index = get_bin_label_holder_index(rules)
    check_bin_rule_pb_valid(public_info, bins)
    if check_bin_rule_is_woe(label_holder_index):
        return woe_bin_rule_from_pb_and_old_rule(rules, bins)
    else:
        return normal_bin_rule_from_pb_and_old_rule(rules, bins)


def check_bin_rule_pb_valid(public_info: dict, bins: Bins):
    assert bins.model_hash == public_info.get(
        "model_hash", ""
    ), "model hash mismatch, make sure the pb is matched to the original model"
