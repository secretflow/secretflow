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
from typing import Dict, List

import numpy as np

import secretflow.component.ml.linear.ss_glm as glm
import secretflow.component.ml.linear.ss_sgd as sgd
from secretflow.component.data_utils import (
    DistDataType,
    extract_table_header,
    model_loads,
)
from secretflow.device import PYU, SPU, PYUObject, SPUObject
from secretflow.ml.linear import RegType
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.utils.sigmoid import SigType

from .graph_builder_manager import GraphBuilderManager

SF_TYPE_TO_SERVING_TYPE = {
    np.int8: 'DT_INT8',
    np.int16: 'DT_INT16',
    np.int32: 'DT_INT32',
    np.int64: 'DT_INT64',
    np.uint8: 'DT_UINT8',
    np.uint16: 'DT_UINT16',
    np.uint32: 'DT_UINT32',
    np.uint64: 'DT_UINT64',
    np.float32: 'DT_FLOAT',
    np.float64: 'DT_DOUBLE',
    bool: 'DT_BOOL',
    int: 'DT_INT64',
    float: 'DT_DOUBLE',
    object: 'DT_STRING',
}

SS_GLM_LINK_MAP = {
    "Logit": "LF_SIGMOID_SR",
    "Log": "LF_LOG",
    "Reciprocal": "LF_RECIPROCAL",
    "Identity": "LF_IDENTITY",
}


SS_SGD_LINK_MAP = {
    SigType.REAL: "LF_SIGMOID_RAW",
    SigType.T1: "LF_SIGMOID_T1",
    SigType.T3: "LF_SIGMOID_T3",
    SigType.T5: "LF_SIGMOID_T5",
    SigType.DF: "LF_SIGMOID_DF",
    SigType.SR: "LF_SIGMOID_SR",
    SigType.MIX: "LF_SIGMOID_SEGLS",
}


def sf_type_to_serving_str(t: np.dtype) -> str:
    assert t in SF_TYPE_TO_SERVING_TYPE, f"not support type {t}"
    return SF_TYPE_TO_SERVING_TYPE[t]


def reveal_to_pyu(
    spu: SPU, spu_w: SPUObject, start: int, end: int, to: PYU
) -> PYUObject:
    def _slice(w):
        w = w.reshape((-1, 1))
        return w.flatten()[start:end]

    sliced_w = spu(_slice)(spu_w)
    return to(lambda w: list(w))(sliced_w.to(to))


def linear_model_converter(
    party_features_length: Dict[str, int],
    input_schema: Dict[str, Dict[str, np.dtype]],
    feature_names: List[str],
    spu_w: SPUObject,
    label_col: str,
    offset_col: str,
    yhat_scale: float,
    link_type: str,
    pred_name: str,
):
    assert set(party_features_length) == set(input_schema)
    spu = spu_w.device
    party_pos = 0
    party_dot_kwargs = dict()
    party_merge_kwargs = dict()
    for party, f_len in party_features_length.items():
        assert party in input_schema
        pyu = PYU(party)

        input_features = input_schema[party]
        party_features = feature_names[party_pos : party_pos + f_len]
        assert set(party_features).issubset(set(input_features))
        pyu_w = reveal_to_pyu(spu, spu_w, party_pos, party_pos + f_len, pyu)
        party_pos += f_len

        if offset_col in input_features:
            party_features.append(offset_col)
            pyu_w = pyu(lambda w: w.append(1.0))(pyu_w)

        if label_col in input_features:
            intercept = reveal_to_pyu(
                spu, spu_w, len(feature_names), len(feature_names) + 1, pyu
            )
            intercept = pyu(lambda i: i[0])(intercept)
        else:
            intercept = 0

        party_dot_kwargs[pyu] = {
            "feature_names": party_features,
            "feature_weights": pyu_w,
            "input_types": [
                sf_type_to_serving_str(input_features[f]) for f in party_features
            ],
            "output_col_name": "partial_y",
            "intercept": intercept,
        }

        party_merge_kwargs[pyu] = {
            "yhat_scale": yhat_scale,
            "link_function": link_type,
            "input_col_name": "partial_y",
            "output_col_name": pred_name,
        }

    return party_dot_kwargs, party_merge_kwargs


def ss_glm_converter(
    ctx,
    node_id: int,
    builder: GraphBuilderManager,
    spu_config,
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
    pred_name: str,
):
    cluster_def = spu_config["cluster_def"].copy()
    cluster_def["runtime_config"]["field"] = "FM128"
    cluster_def["runtime_config"]["fxp_fraction_bits"] = 40
    spu = SPU(cluster_def, spu_config["link_desc"])

    model_objs, model_meta_str = model_loads(
        ctx,
        model_ds,
        glm.MODEL_MAX_MAJOR_VERSION,
        glm.MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_GLM_MODEL,
        spu=spu,
    )
    meta = json.loads(model_meta_str)
    spu_w = model_objs[0]

    feature_names = meta["feature_names"]
    party_features_length: Dict[str, int] = meta["party_features_length"]
    offset_col = meta["offset_col"]
    label_col = meta["label_col"]
    yhat_scale = meta["y_scale"]
    assert len(label_col) == 1
    assert meta["link"] in SS_GLM_LINK_MAP
    label_col = label_col[0]

    party_dot_kwargs, party_merge_kwargs = linear_model_converter(
        party_features_length,
        input_schema,
        feature_names,
        spu_w,
        label_col,
        offset_col,
        yhat_scale,
        SS_GLM_LINK_MAP[meta["link"]],
        pred_name,
    )

    builder.add_node(f"ss_glm_{node_id}_dot", "dot_product", party_dot_kwargs)
    builder.new_execution("DP_ANYONE")
    builder.add_node(f"ss_glm_{node_id}_merge_y", "merge_y", party_merge_kwargs)


def ss_sgd_converter(
    ctx,
    node_id: int,
    builder: GraphBuilderManager,
    spu_config: SPU,
    input_schema: Dict[str, Dict[str, np.dtype]],
    model_ds: DistData,
    pred_name: str,
):
    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    model_objs, model_meta_str = model_loads(
        ctx,
        model_ds,
        sgd.MODEL_MAX_MAJOR_VERSION,
        sgd.MODEL_MAX_MINOR_VERSION,
        DistDataType.SS_SGD_MODEL,
        spu=spu,
    )
    meta = json.loads(model_meta_str)
    spu_w = model_objs[0]

    feature_names = meta["feature_selects"]
    party_features_length: Dict[str, int] = meta["party_features_length"]
    reg_type = RegType(meta["reg_type"])
    sig_type = SigType(meta["sig_type"])
    label_col = meta["label_col"]
    assert len(label_col) == 1
    label_col = label_col[0]

    if reg_type == RegType.Linear:
        link_type = "LF_IDENTITY"
    else:
        link_type = SS_SGD_LINK_MAP[sig_type]

    party_dot_kwargs, party_merge_kwargs = linear_model_converter(
        party_features_length,
        input_schema,
        feature_names,
        spu_w,
        label_col,
        None,
        1.0,
        link_type,
        pred_name,
    )

    builder.add_node(f"ss_glm_{node_id}_dot", "dot_product", party_dot_kwargs)
    builder.new_execution("DP_ANYONE")
    builder.add_node(f"ss_glm_{node_id}_merge_y", "merge_y", party_merge_kwargs)


def train_model_converter(
    ctx,
    builder: GraphBuilderManager,
    node_id: int,
    spu_config,
    param: NodeEvalParam,
    in_ds: List[DistData],
    out_ds: List[DistData],
):
    in_dataset = [d for d in in_ds if d.type == DistDataType.VERTICAL_TABLE]
    assert len(in_dataset) == 1
    in_dataset = in_dataset[0]

    input_schema = extract_table_header(
        in_dataset, load_features=True, load_ids=True, load_labels=True
    )

    assert set(input_schema) == set([p.party for p in builder.pyus])

    if param.name in ["ss_glm_predict", "ss_glm_train"]:
        # SS_GLM_MODEL
        if param.name == "ss_glm_train":
            model_ds = [d for d in out_ds if d.type == DistDataType.SS_GLM_MODEL]
        else:
            model_ds = [d for d in in_ds if d.type == DistDataType.SS_GLM_MODEL]
        assert len(model_ds) == 1
        model_ds = model_ds[0]
        ss_glm_converter(
            ctx,
            node_id,
            builder,
            spu_config,
            input_schema,
            model_ds,
            "pred_y",
        )
    elif param.name in ["ss_sgd_train", "ss_sgd_predict"]:
        # SS_SGD_MODEL
        if param.name == "ss_sgd_train":
            model_ds = [d for d in out_ds if d.type == DistDataType.SS_SGD_MODEL]
        else:
            model_ds = [d for d in in_ds if d.type == DistDataType.SS_SGD_MODEL]
        assert len(model_ds) == 1
        model_ds = model_ds[0]
        ss_sgd_converter(
            ctx,
            node_id,
            builder,
            spu_config,
            input_schema,
            model_ds,
            "pred_y",
        )
    else:
        # TODO others model
        raise AttributeError(f"not support param.name {param.name}")
