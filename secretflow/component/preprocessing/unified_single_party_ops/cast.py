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

import pyarrow as pa
import pyarrow.compute as pc

import secretflow.compute as sc
from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    ServingBuilder,
    UnionSelection,
    VTable,
    register,
)

from ..preprocessing import PreprocessingMixin


@register(domain='preprocessing', version='1.0.0')
class Cast(PreprocessingMixin, Component):
    '''
    For conversion between basic data types, such as converting float to string.
    '''

    astype: str = Field.union_attr(
        desc="single-choice, options available are string, integer, float",
        selections=[
            UnionSelection('integer', 'integer'),
            UnionSelection('float', 'float'),
            UnionSelection('string', 'string'),
        ],
    )
    columns: list[str] = Field.table_column_attr(
        "input_ds",
        desc="Multiple-choice, options available are string, integer, float, boolean",
        limit=Interval.closed(1, None),
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="The input table",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="The output table",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_rule: Output = Field.output(
        desc="The output rules",
        types=[DistDataType.PREPROCESSING_RULE],
    )

    @staticmethod
    def apply(table: sc.Table, target: str) -> sc.Table:
        is_numeric_target = target == 'int' or target == 'float'
        if target == 'int':
            pa_type = pa.int64()
        elif target == 'float':
            pa_type = pa.float64()
        elif target == 'str':
            pa_type = pa.string()
        else:
            raise ValueError(f"unsupported target type {target}")

        columns = table.column_names
        for col_name in columns:
            try:
                col = table.column(col_name)
                if col.dtype == pa_type:
                    new_col = col
                else:
                    if pa.types.is_string(col.dtype) and is_numeric_target:
                        col = sc.utf8_trim(col, characters=" \t\n\v\f\r\"'")
                    options = pc.CastOptions.safe(pa_type)
                    new_col = sc.cast(col, options=options)
                table = table.set_column(
                    table.column_names.index(col_name), col_name, new_col
                )
            except Exception as e:
                raise ValueError(f"cast {col_name} failed, {e}")
        return table

    def evaluate(self, ctx: Context):
        astype_map = {
            "integer": "int",
            "float": "float",
            "string": "str",
        }

        astype = astype_map[self.astype]

        def _fit(trans_data: sc.Table) -> sc.Table:
            table = Cast.apply(trans_data, astype)
            return table

        in_tbl = VTable.from_distdata(self.input_ds)
        tran_tbl = in_tbl.select(self.columns)
        rule = self.fit(ctx, self.output_rule, tran_tbl, _fit)
        self.transform(ctx, self.output_ds, in_tbl, rule)

    def export(self, ctx: Context, builder: ServingBuilder) -> None:
        self.do_export(ctx, builder, self.input_ds, self.output_rule.data)
