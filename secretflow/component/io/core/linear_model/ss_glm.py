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
import json
import logging
from typing import List

import numpy as np

from secretflow.device import SPUObject
from secretflow.device.driver import reveal
from secretflow.spec.extend.linear_model_pb2 import FeatureWeight, LinearModel


def ss_glm_to_linear_model_pb(
    ss_glm_model: List[SPUObject], public_info: str
) -> LinearModel:
    """
    WARNING: THIS MODEL IS NOT SAFE, ALL MODEL INFO IS REVEALED AND LABEL INFO IS AT GREAT RISK
    DO NOT USE THIS UNLESS YOU UNDERSTAND THIS RISK.
    Convert SSGLM to Linear Model.
    ss_glm_model: list of SPUObject represents glm model.
    public_info: str, model meta in json format
    """
    logging.warning(
        "DANGER! WARNING! WE ARE REVEALING SS GLM MODEL. ALL MODEL INFO IS REVEALED. THE LABEL INFO IS AT GREAT RISK!"
    )
    w: np.ndarray = reveal(ss_glm_model[0]).reshape((-1,))
    info_dict = json.loads(public_info)
    feature_names = info_dict["feature_names"]
    party_features_length = info_dict["party_features_length"]
    feature_number = len(feature_names)

    assert (feature_number + 1) == w.size

    linear_model = LinearModel()
    party_belongings = []
    for party, length in party_features_length.items():
        party_belongings.extend([party] * length)

    assert feature_number == len(party_belongings)

    for i in range(feature_number):
        fw = FeatureWeight()
        fw.party = party_belongings[i]
        fw.feature_name = feature_names[i]
        fw.feature_weight = w[i]
        linear_model.feature_weights.append(fw)

    linear_model.bias = w[-1]
    linear_model.model_hash = json.loads(public_info).get("model_hash", "")
    return linear_model


def ss_glm_from_pb_and_old_model(
    ss_glm_model: List[SPUObject],
    original_public_info: str,
    linear_model_pb: LinearModel,
) -> List[SPUObject]:
    assert len(ss_glm_model) == 1, "assumed ss glm model structure has len 1"
    check_pb_match_old_model(original_public_info, linear_model_pb)
    spu = ss_glm_model[0].device
    glm_model_size = reveal(spu(lambda x: x.size)(ss_glm_model[0]))
    w_length = len(linear_model_pb.feature_weights) + 1
    assert w_length == glm_model_size
    new_w = np.zeros((w_length, 1))
    for i in range(w_length - 1):
        new_w[i, 0] = linear_model_pb.feature_weights[i].feature_weight
    new_w[-1, 0] = linear_model_pb.bias
    return [spu(lambda: new_w)()]


def check_pb_match_old_model(original_public_info: str, linear_model_pb: LinearModel):
    info_dict = json.loads(original_public_info)
    assert (
        info_dict.get("model_hash", "") == linear_model_pb.model_hash
    ), f"model hash mismatch {info_dict['model_hash']}, {linear_model_pb.model_hash}"
    feature_names = info_dict["feature_names"]
    party_features_length = info_dict["party_features_length"]
    feature_number = len(feature_names)

    party_belongings = []
    for party, length in party_features_length.items():
        party_belongings.extend([party] * length)

    # check feature number equal
    assert feature_number == len(
        linear_model_pb.feature_weights
    ), f"feature number mismatch, original feature number is {feature_number}, \
        linear_model_pb feature number is {len(linear_model_pb.feature_weights)}, \
        make sure the linear model pb is matched to original model."
    for i in range(feature_number):
        # check feature order and name equal
        assert (
            feature_names[i] == linear_model_pb.feature_weights[i].feature_name
        ), f"feature name mismatch, original feature name is {feature_names[i]}, \
            linear_model_pb feature name is {linear_model_pb.feature_weights[i].feature_name}, \
            make sure the linear model pb is matched to original model."
        # check feature belongings equal
        assert (
            party_belongings[i] == linear_model_pb.feature_weights[i].party
        ), f"feature party mismatch, original feature party is {party_belongings[i]}, \
            linear_model_pb feature party is {linear_model_pb.feature_weights[i].party}, \
            make sure the linear model pb is matched to original model."
