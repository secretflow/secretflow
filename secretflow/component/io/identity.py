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


from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    register,
)

IDENTITY_SUPPORTED_TYPES = [
    DistDataType.SS_GLM_MODEL,
    DistDataType.SGB_MODEL,
    DistDataType.SS_XGB_MODEL,
    DistDataType.SS_SGD_MODEL,
    DistDataType.BINNING_RULE,
    DistDataType.READ_DATA,
]


@register(domain="io", version="1.0.0")
class Identity(Component):
    '''
    map any input to output
    '''

    input_data: Input = Field.input(  # type: ignore
        desc="Input data",
        types=IDENTITY_SUPPORTED_TYPES,
    )
    output_data: Output = Field.output(
        desc="Output data",
        types=IDENTITY_SUPPORTED_TYPES,
    )

    def evaluate(self, ctx: Context):
        spu = ctx.make_spu()
        model = ctx.load_model(self.input_data, spu=spu)
        ctx.dump_to(model, self.output_data)
