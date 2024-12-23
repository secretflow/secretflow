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


import secretflow.compute as sc
from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    ServingBuilder,
    VTable,
    VTableField,
    VTableFieldKind,
    register,
)

from ..preprocessing import PreprocessingMixin

OP_MAP = {
    "+": sc.add,
    "-": sc.subtract,
    "*": sc.multiply,
    "/": sc.divide,
}


@register(domain="preprocessing", version="1.0.0")
class BinaryOp(PreprocessingMixin, Component):
    '''
    Perform binary operation binary_op(f1, f2) and assign the result to f3, f3 can be new or old. Currently f1, f2 and f3 all belong to a single party.
    '''

    binary_op: str = Field.attr(
        desc="What kind of binary operation we want to do, currently only supports +, -, *, /",
        default="+",
        choices=["+", "-", "*", "/"],
    )
    new_feature_name: str = Field.attr(
        desc="Name of the newly generated feature.",
    )
    as_label: bool = Field.attr(
        desc="If True, the generated feature will be marked as label in schema.",
        default=False,
    )
    f1: str = Field.table_column_attr(
        "input_ds",
        desc="Feature 1 to operate on.",
    )
    f2: str = Field.table_column_attr(
        "input_ds",
        desc="Feature 2 to operate on.",
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_rule: Output = Field.output(
        desc="feature gen rule",
        types=[DistDataType.PREPROCESSING_RULE],
    )

    def evaluate(self, ctx: Context):
        in_tbl = VTable.from_distdata(self.input_ds)
        load_columns = [self.f1, self.f2]
        is_new_column = True
        for p in in_tbl.parties.values():
            if self.new_feature_name in p.schema.fields:
                is_new_column = False
                if self.new_feature_name not in load_columns:
                    load_columns.append(self.new_feature_name)
                break

        def _fit(df: sc.Table) -> sc.Table:
            arg_0 = df.column(self.f1)
            arg_1 = df.column(self.f2)
            new_col = OP_MAP[self.binary_op](arg_0, arg_1)

            if not is_new_column:
                df = df.set_column(
                    df.column_names.index(self.new_feature_name),
                    self.new_feature_name,
                    new_col,
                )
            else:
                kind = VTableFieldKind.LABEL if self.as_label else VTableFieldKind.ID
                field = VTableField.pa_field(self.new_feature_name, new_col.dtype, kind)
                df = df.append_column(field, new_col)

            return df

        trans_tbl = in_tbl.select(load_columns)
        if len(trans_tbl.parties) != 1:
            raise ValueError(
                f'{load_columns} must belong to a single party, but got {trans_tbl.parties.keys()}'
            )
        rule = self.fit(ctx, self.output_rule, trans_tbl, _fit)
        self.transform(ctx, self.output_ds, in_tbl, rule)

    def export(self, ctx: Context, builder: ServingBuilder) -> None:
        self.do_export(ctx, builder, self.input_ds, self.output_rule.data)
