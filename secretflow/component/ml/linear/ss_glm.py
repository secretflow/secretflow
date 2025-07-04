# Copyright 2024 Ant Group Co., Ltd.
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

from secretflow_serving_lib.link_function_pb2 import LinkFunctionType
from secretflow_spec.v1.data_pb2 import DistData

from secretflow.component.core import (
    SPU_RUNTIME_CONFIG_FM128_FXP40,
    SS_GLM_MODEL_MAX,
    Context,
    DistDataType,
    ServingBuilder,
    VTable,
)

from .linear import build_linear_model, build_phe_linear_model, get_party_features_info

SS_GLM_LINK_MAP = {
    "Logit": LinkFunctionType.LF_SIGMOID_SR,
    "Log": LinkFunctionType.LF_EXP,
    "Reciprocal": LinkFunctionType.LF_RECIPROCAL,
    "Identity": LinkFunctionType.LF_IDENTITY,
}


class SSGLMExportMixin:
    def do_export(
        self,
        ctx: Context,
        builder: ServingBuilder,
        input_dd: DistData,
        model_dd: DistData,
        he_mode: bool,
    ):
        input_tbl = VTable.from_distdata(input_dd)
        spu = ctx.make_spu(config=SPU_RUNTIME_CONFIG_FM128_FXP40)
        model = ctx.load_model(
            model_dd, DistDataType.SS_GLM_MODEL, SS_GLM_MODEL_MAX, spu=spu
        )
        meta = json.loads(model.public_info)
        spu_w = model.objs[0]

        feature_names = meta["feature_names"]
        party_features_name, party_features_pos = get_party_features_info(meta)
        offset_col = meta["offset_col"]
        if offset_col:
            assert len(offset_col) == 1
            offset_col = offset_col[0]
        else:
            offset_col = ""
        label_col = meta["label_col"]
        yhat_scale = meta["y_scale"]
        assert len(label_col) == 1
        assert meta["link"] in SS_GLM_LINK_MAP
        label_col = label_col[0]

        link_type = SS_GLM_LINK_MAP[meta["link"]]
        if link_type == LinkFunctionType.LF_EXP and meta["fxp_exp_mode"] == 2:
            link_type = LinkFunctionType.LF_EXP_TAYLOR
            exp_iters = meta["fxp_exp_iters"]
        else:
            exp_iters = 0

        node_prefix = f"ss_glm_{builder.max_id()}"
        if he_mode:
            heu_dict = ctx.make_heus(
                input_tbl.parties.keys(), spu.conf.field, spu.conf.fxp_fraction_bits
            )
            builder.set_he_config(heu_dict)
            build_phe_linear_model(
                builder,
                node_prefix,
                heu_dict,
                party_features_name,
                party_features_pos,
                input_tbl.schemas,
                feature_names,
                spu_w,
                label_col,
                offset_col,
                yhat_scale,
                link_type,
                exp_iters,
            )
        else:
            build_linear_model(
                builder,
                node_prefix,
                party_features_name,
                party_features_pos,
                input_tbl.schemas,
                feature_names,
                spu_w,
                label_col,
                offset_col,
                yhat_scale,
                link_type,
                exp_iters,
            )
