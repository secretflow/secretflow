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

from google.protobuf.json_format import Parse

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
    Model,
    Output,
    Registry,
    register,
    uuid4,
)
from secretflow.component.io.core.bins.bins import bin_rule_from_pb_and_old_rule
from secretflow.component.io.core.boost_model.sgb import get_sgb_snapshot_from_pb
from secretflow.component.io.core.linear_model.ss_glm import (
    ss_glm_from_pb,
    ss_glm_from_pb_and_old_model,
)
from secretflow.component.preprocessing.binning.vert_binning import VertBinning
from secretflow.component.preprocessing.binning.vert_woe_binning import VertWoeBinning
from secretflow.component.preprocessing.preprocessing import PreprocessingMixin
from secretflow.device import PYU
from secretflow.error_system.exceptions import CompEvalError
from secretflow.spec.extend.bin_data_pb2 import Bins
from secretflow.spec.extend.linear_model_pb2 import GeneralizedLinearModel, LinearModel

CURRENT_SUPPORTED_TYPES = [
    str(DistDataType.BINNING_RULE),
    str(DistDataType.SS_GLM_MODEL),
    str(DistDataType.SGB_MODEL),
]


@register(domain="io", version="1.0.0")
class WriteData(Component):
    '''
    write model or rules back to sf cluster
    '''

    write_data: str = Field.attr(
        desc="rule or model protobuf by json format",
    )
    write_data_type: str = Field.attr(
        desc="which rule or model is writing",
        default=CURRENT_SUPPORTED_TYPES[0],
        choices=CURRENT_SUPPORTED_TYPES,
    )
    input_data: Input = Field.input(  # type: ignore
        desc="Input dist data. Rule reconstructions may need hidden info in original rule for security considerations.",
        types=[
            DistDataType.BINNING_RULE,
            DistDataType.SS_GLM_MODEL,
            DistDataType.SGB_MODEL,
            DistDataType.NULL,
        ],
    )
    output_data: Output = Field.output(
        desc="Output rules or models in sf cluster format",
        types=[
            DistDataType.BINNING_RULE,
            DistDataType.SS_GLM_MODEL,
            DistDataType.SGB_MODEL,
            # add others module or rules support here
        ],
    )

    def evaluate(self, ctx: Context):
        if self.write_data_type == str(DistDataType.BINNING_RULE):
            model = ctx.load_model(
                self.input_data, DistDataType.BINNING_RULE, BINNING_RULE_MAX
            )
            bin_rules = Parse(self.write_data, Bins())
            rule_objs, is_woe = bin_rule_from_pb_and_old_rule(
                model.objs, model.metadata, bin_rules
            )

            cls = VertWoeBinning if is_woe else VertBinning
            defi = Registry.get_definition_by_class(cls)
            out_model = PreprocessingMixin.build_model(
                defi.component_id,
                DistDataType.BINNING_RULE,
                BINNING_RULE_MAX,
                rule_objs,
                model.public_info,
                system_info=self.input_data.system_info,
            )

            ctx.dump_to(out_model, self.output_data)
        elif self.write_data_type == str(DistDataType.SS_GLM_MODEL):
            spu = ctx.make_spu(config=SPU_RUNTIME_CONFIG_FM128_FXP40)

            if self.input_data is None:
                try:
                    glm_pb = Parse(self.write_data, GeneralizedLinearModel())
                except:
                    raise CompEvalError(
                        "write_data: {write_data} is not a valid GeneralizedLinearModel protobuf."
                    )
                pyu = PYU(ctx.parties[0])
                new_model_objs, public_info = ss_glm_from_pb(spu, pyu, glm_pb)
                system_info = None
            else:
                model = ctx.load_model(
                    self.input_data, DistDataType.SS_GLM_MODEL, GLM_MODEL_MAX, spu=spu
                )
                try:
                    lm_pb = Parse(self.write_data, LinearModel())
                except:
                    raise CompEvalError(
                        "write_data: {write_data} is not a valid LinearModel protobuf."
                    )
                pyu = PYU(ctx.parties[0])
                new_model_objs = ss_glm_from_pb_and_old_model(
                    model.objs, model.public_info, lm_pb, pyu
                )
                public_info = model.public_info
                system_info = self.input_data.system_info
            info_dict = json.loads(public_info)
            info_dict["model_hash"] = uuid4(pyu)

            out_model = Model(
                name="ss_glm",
                type=DistDataType.SS_GLM_MODEL,
                version=GLM_MODEL_MAX,
                objs=new_model_objs,
                public_info=json.dumps(info_dict),
                system_info=system_info,
            )
            ctx.dump_to(out_model, self.output_data)
        elif self.write_data_type == str(DistDataType.SGB_MODEL):
            snapshot = get_sgb_snapshot_from_pb(self.write_data)
            out_model = Model(
                name="sgb",
                type=DistDataType.SGB_MODEL,
                version=SGB_MODEL_MAX,
                objs=snapshot.model_objs,
                public_info=json.dumps(snapshot.model_meta),
                system_info=None,
            )
            ctx.dump_to(out_model, self.output_data)
        else:
            raise AttributeError(f"unknown model/rules type {self.write_data_type}")
