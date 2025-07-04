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


from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    IServingExporter,
    Output,
    ServingBuilder,
    VTable,
    register,
)
from secretflow.device import PYU

from ..preprocessing import PreprocessingMixin


@register(domain='preprocessing', version='1.0.0')
class Substitution(PreprocessingMixin, Component, IServingExporter):
    '''
    unified substitution component
    '''

    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    input_rule: Input = Field.input(
        desc="Input preprocessing rules",
        types=[DistDataType.PREPROCESSING_RULE, DistDataType.BINNING_RULE],
    )
    output_ds: Output = Field.output(
        desc="output_dataset",
        types=[DistDataType.VERTICAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        version_max = self.get_version_max(self.input_rule.type)
        t = VTable.from_distdata(self.input_ds)
        pyus = {p: PYU(p) for p in t.parties.keys()}
        model = ctx.load_model(self.input_rule, version=version_max, pyus=pyus)
        self.transform(ctx, self.output_ds, self.input_ds, model)

    def export(self, ctx: Context, builder: ServingBuilder):
        self.do_export(
            ctx, builder, self.input_ds, self.input_rule, is_substitution=True
        )
