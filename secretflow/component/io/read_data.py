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

from secretflow.component.core import (
    BINNING_RULE_MAX,
    GLM_MODEL_MAX,
    SGB_MODEL_MAX,
    SPU_RUNTIME_CONFIG_FM128_FXP40,
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    register,
)
from secretflow.component.io.core.bins.bins import bin_rule_to_pb
from secretflow.component.io.core.boost_model.sgb import sgb_model_to_pb
from secretflow.component.io.core.linear_model.ss_glm import (
    ss_glm_to_generalized_linear_model_pb,
    ss_glm_to_linear_model_pb,
)
from secretflow.device import PYU
from secretflow.ml.boost.sgb_v.checkpoint import SGBSnapshot, build_sgb_model
from secretflow.spec.v1.data_pb2 import DistData


@register(domain="io", version="1.0.0")
class ReadData(Component):
    '''
    read model or rules from sf cluster
    '''

    generalized_linear_model: bool = Field.attr(
        desc="Whether to dump the complete generalized linear model. The complete generalized linear model contains link, y_scale, offset_col, and so on.",
        default=False,
    )
    input_data: Input = Field.input(  # type: ignore
        desc="Input dist data",
        types=[
            DistDataType.BINNING_RULE,
            DistDataType.SS_GLM_MODEL,
            DistDataType.SGB_MODEL,
        ],
    )
    output_data: Output = Field.output(
        desc="Output rules or models in DistData.meta",
        types=[DistDataType.READ_DATA],
    )

    def evaluate(self, ctx: Context):
        input_type = self.input_data.type
        if input_type == DistDataType.BINNING_RULE:
            model = ctx.load_model(
                self.input_data, DistDataType.BINNING_RULE, BINNING_RULE_MAX
            )

            read_data = bin_rule_to_pb(model.objs, model.metadata)
        elif input_type == DistDataType.SS_GLM_MODEL:
            spu = ctx.make_spu(config=SPU_RUNTIME_CONFIG_FM128_FXP40)
            model = ctx.load_model(
                self.input_data, DistDataType.SS_GLM_MODEL, GLM_MODEL_MAX, spu=spu
            )
            if self.generalized_linear_model:
                read_data = ss_glm_to_generalized_linear_model_pb(
                    model.objs, model.public_info
                )
            else:
                read_data = ss_glm_to_linear_model_pb(model.objs, model.public_info)
        elif input_type == DistDataType.SGB_MODEL:
            pyus = {p: PYU(p) for p in ctx.parties}
            model = ctx.load_model(
                self.input_data, DistDataType.SGB_MODEL, SGB_MODEL_MAX, pyus=pyus
            )
            snap_shot = SGBSnapshot(model.objs, json.loads(model.public_info))
            sgb_model = build_sgb_model(snap_shot)
            read_data = sgb_model_to_pb(sgb_model, snap_shot.model_meta)
        else:
            raise AttributeError(f"unknown model/rules type {input_type}")

        output_data_dd = DistData(
            name=self.output_data.uri,
            type=str(DistDataType.READ_DATA),
            system_info=self.input_data.system_info,
        )
        output_data_dd.meta.Pack(read_data)
        self.output_data.data = output_data_dd
