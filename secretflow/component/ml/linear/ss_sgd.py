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

from secretflow.component.core import (
    SS_SGD_MODEL_MAX,
    Context,
    DistDataType,
    ServingBuilder,
    VTable,
)
from secretflow.error_system.exceptions import (
    CompEvalError,
    DataFormatError,
    SFModelError,
)
from secretflow.ml.linear import RegType
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.utils.sigmoid import SigType

from .linear import build_linear_model, build_phe_linear_model, get_party_features_info

SS_SGD_LINK_MAP = {
    SigType.REAL: LinkFunctionType.LF_SIGMOID_RAW,
    SigType.T1: LinkFunctionType.LF_SIGMOID_T1,
    SigType.T3: LinkFunctionType.LF_SIGMOID_T3,
    SigType.T5: LinkFunctionType.LF_SIGMOID_T5,
    SigType.DF: LinkFunctionType.LF_SIGMOID_DF,
    SigType.SR: LinkFunctionType.LF_SIGMOID_SR,
    SigType.MIX: LinkFunctionType.LF_SIGMOID_SEGLS,
}


class SSSGDExportMixin:
    def do_export(
        self,
        ctx: Context,
        builder: ServingBuilder,
        input_dd: DistData,
        model_dd: DistData,
        he_mode: bool,
    ):
        input_tbl = VTable.from_distdata(input_dd)
        spu = ctx.make_spu()
        model = ctx.load_model(
            model_dd, DistDataType.SS_SGD_MODEL, SS_SGD_MODEL_MAX, spu=spu
        )
        meta = json.loads(model.public_info)
        spu_w = model.objs[0]

        feature_names = meta["feature_names"]
        party_features_name, party_features_pos = get_party_features_info(meta)
        reg_type = RegType(meta["reg_type"])
        sig_type = SigType(meta["sig_type"])
        label_col = meta["label_col"]
        assert len(label_col) == 1
        label_col = label_col[0]

        if reg_type == RegType.Linear:
            link_type = LinkFunctionType.LF_IDENTITY
        else:
            link_type = SS_SGD_LINK_MAP[sig_type]

        node_prefix = f"ss_sgd_{builder.max_id()}"
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
                None,
                1.0,
                link_type,
                0,
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
                None,
                1.0,
                link_type,
                0,
            )
