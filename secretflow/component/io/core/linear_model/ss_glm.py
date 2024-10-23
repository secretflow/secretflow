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
import json
import logging
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from secretflow.error_system.exceptions import CompEvalError
from secretflow.device import SPUObject
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU
from secretflow.device.driver import reveal
from secretflow.error_system.exceptions import (
    DataFormatError,
    SFModelError,
)
from secretflow.spec.extend.linear_model_pb2 import (
    FeatureWeight,
    GeneralizedLinearModel,
    LinearModel,
    PublicInfo,
)


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

    if (feature_number + 1) != w.size:
        raise DataFormatError.feature_not_matched(
            f"feature number {feature_number} + 1 is not equal to ss_glm_model shape's size {w.size}"
        )

    linear_model = LinearModel()
    party_belongings = []
    for party, length in party_features_length.items():
        party_belongings.extend([party] * length)

    if feature_number != len(party_belongings):
        raise DataFormatError.feature_not_matched(
            f"feature number {feature_number} is not equal to party_belongings length {len(party_belongings)}"
        )

    for i in range(feature_number):
        fw = FeatureWeight()
        fw.party = party_belongings[i]
        fw.feature_name = feature_names[i]
        fw.feature_weight = w[i]
        linear_model.feature_weights.append(fw)

    linear_model.bias = w[-1]
    linear_model.model_hash = json.loads(public_info).get("model_hash", "")
    return linear_model


def ss_glm_to_generalized_linear_model_pb(
    ss_glm_model: List[SPUObject], public_info: str
) -> GeneralizedLinearModel:
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
    linear_model_pb = ss_glm_to_linear_model_pb(ss_glm_model, public_info)

    public_info = json.loads(public_info)
    public_info_pb = PublicInfo()
    public_info_pb.link = public_info['link']
    public_info_pb.y_scale = public_info['y_scale']

    offset_col = public_info['offset_col']
    if offset_col:
        assert len(offset_col) == 1
        offset_col = offset_col[0]
    else:
        offset_col = ""
    public_info_pb.offset_col = offset_col

    label_col = public_info['label_col']
    if label_col:
        assert len(label_col) == 1
        label_col = label_col[0]
    else:
        label_col = ""
    public_info_pb.label_col = label_col

    public_info_pb.fxp_exp_mode = public_info['fxp_exp_mode']
    public_info_pb.fxp_exp_iters = public_info['fxp_exp_iters']

    generalized_linear_model = GeneralizedLinearModel()
    generalized_linear_model.model.CopyFrom(linear_model_pb)
    generalized_linear_model.public_info.CopyFrom(public_info_pb)
    return generalized_linear_model


def ss_glm_from_pb_and_old_model(
    ss_glm_model: List[SPUObject],
    original_public_info: str,
    linear_model_pb: LinearModel,
    pyu: PYU,
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
    return [pyu(lambda: new_w)().to(spu)]


def check_pb_match_old_model(original_public_info: str, linear_model_pb: LinearModel):
    info_dict = json.loads(original_public_info)
    if info_dict.get("model_hash", "") != linear_model_pb.model_hash:
        raise SFModelError.model_hash_mismatch(
            f"model hash mismatch {info_dict['model_hash']}, {linear_model_pb.model_hash}"
        )
    feature_names = info_dict["feature_names"]
    party_features_length = info_dict["party_features_length"]
    feature_number = len(feature_names)

    party_belongings = []
    for party, length in party_features_length.items():
        party_belongings.extend([party] * length)

    # check feature number equal
    if feature_number != len(linear_model_pb.feature_weights):
        raise DataFormatError.feature_not_matched(
            f"feature number mismatch, original feature number is {feature_number},"
            f"linear_model_pb feature number is {len(linear_model_pb.feature_weights)},"
            "make sure the linear model pb is matched to original model."
        )
    for i in range(feature_number):
        # check feature order and name equal
        if feature_names[i] != linear_model_pb.feature_weights[i].feature_name:
            raise DataFormatError.feature_not_matched(
                f"feature name mismatch, original feature name is {feature_names[i]},"
                f"linear_model_pb feature name is {linear_model_pb.feature_weights[i].feature_name},"
                "make sure the linear model pb is matched to original model."
            )
        # check feature belongings equal
        if party_belongings[i] != linear_model_pb.feature_weights[i].party:
            raise DataFormatError.feature_not_matched(
                f"feature party mismatch, original feature party is {party_belongings[i]},"
                f"linear_model_pb feature party is {linear_model_pb.feature_weights[i].party},"
                "make sure the linear model pb is matched to original model."
            )


def ss_glm_from_pb(
    spu: SPU,
    pyu: PYU,
    generalized_linear_model_pb: GeneralizedLinearModel,
) -> Tuple[List[SPUObject], str]:
    """
    WARNING: THIS MODEL IS NOT SAFE, ALL MODEL INFO IS REVEALED,
    DO NOT USE THIS UNLESS YOU UNDERSTAND THIS RISK.
    generalized_linear_model_pb: glm model info.
    """
    logging.warning(
        "DANGER! WARNING! ALL MODEL INFO IS REVEALED. THE DATA IS AT RISK! DO NOT USE THIS UNLESS YOU UNDERSTAND THIS RISK!!!"
    )
    w_length = len(generalized_linear_model_pb.model.feature_weights) + 1
    new_w = np.zeros((w_length, 1))
    for i in range(w_length - 1):
        new_w[i, 0] = generalized_linear_model_pb.model.feature_weights[
            i
        ].feature_weight
    new_w[-1, 0] = generalized_linear_model_pb.model.bias

    public_info = {}
    public_info['link'] = generalized_linear_model_pb.public_info.link
    public_info['y_scale'] = generalized_linear_model_pb.public_info.y_scale
    public_info['offset_col'] = generalized_linear_model_pb.public_info.offset_col
    public_info['label_col'] = generalized_linear_model_pb.public_info.label_col
    public_info['fxp_exp_mode'] = generalized_linear_model_pb.public_info.fxp_exp_mode
    public_info['fxp_exp_iters'] = generalized_linear_model_pb.public_info.fxp_exp_iters

    feature_names = []
    party_features_length = defaultdict(int)
    for feature_weight in generalized_linear_model_pb.model.feature_weights:
        feature_names.append(feature_weight.feature_name)
        party_features_length[feature_weight.party] += 1
    non_secure_feature_count = 1
    for party, count in party_features_length.items():
        if count == non_secure_feature_count:
            raise CompEvalError(f"party {party} has 1 feature, which will leak data!!!")
    public_info["feature_names"] = feature_names
    public_info["party_features_length"] = party_features_length

    return [pyu(lambda: new_w)().to(spu)], json.dumps(public_info)
