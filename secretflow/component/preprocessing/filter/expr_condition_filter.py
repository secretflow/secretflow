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


import logging
from typing import Tuple

import duckdb
import pyarrow as pa
from pyarrow import compute as pc

from secretflow.component.core import (
    Component,
    CompVDataFrame,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    VTable,
    register,
    VTableFieldKind,
)
from secretflow.device import PYU, reveal
from secretflow.utils.consistent_ops import unique_list


@register(domain='data_filter', version='1.0.0')
class ExprConditionFilter(Component):
    '''
    Only row-level filtering is supported, column processing is not available;
    the custom expression must comply with SQLite syntax standards
    '''

    # Suggest that the user verify the validity of the expression
    expr: str = Field.attr(
        desc="The custom expression must comply with SQLite syntax standards"
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical or individual table",
        types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output table that satisfies the condition",
        types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    )
    output_ds_else: Output = Field.output(
        desc="Output table that does not satisfies the condition",
        types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    )

    @staticmethod
    def parse_columns(expr: str) -> list[str]:
        import sqlglot
        import sqlglot.expressions as exp

        sql = f"SELECT * FROM __table__ WHERE {expr}"
        columns = []
        for column in sqlglot.parse_one(sql, dialect="duckdb").find_all(exp.Column):
            columns.append(column.sql().strip('"'))

        columns = unique_list(columns)

        return columns

    def evaluate(self, ctx: Context):
        expr = self.expr.strip()
        assert expr != "", f"empty expr"

        columns = self.parse_columns(expr)

        if len(columns) == 0:
            raise AttributeError(f"cannot parse columns from the expr<{expr}>")

        infos = VTable.from_distdata(self.input_ds, columns=columns)
        infos.check_kinds(VTableFieldKind.FEATURE)

        if len(infos.parties) > 1:
            raise AttributeError(
                f"The columns<{columns}> of expr<{expr}> must appears in one party"
            )

        fit_pyu = PYU(infos.party(0).party)

        def _fit(duck_input_table: pa.Table, expr: str) -> Tuple[pa.Table, Exception]:
            sql = f"SELECT CASE WHEN {expr} THEN TRUE ELSE FALSE END AS hit FROM duck_input_table"

            try:
                with duckdb.connect() as con:
                    selection = con.execute(sql).arrow()
                if selection.num_columns != 1 or not pa.types.is_boolean(
                    selection.field(0).type
                ):
                    return None, ValueError(
                        f"The result can only have one column<{selection.num_columns}> and it must be of boolean type<{selection.field(0).type}>"
                    )
                return selection, None
            except duckdb.Error as e:
                logging.exception(f"execute sql {sql} error")
                return None, ValueError(e.__cause__)
            except Exception as e:
                logging.exception(f"execute sql {sql} error")
                return None, e

        reader = CompVDataFrameReader(ctx.storage, ctx.tracer, self.input_ds)

        selected_writer = CompVDataFrameWriter(
            ctx.storage, ctx.tracer, self.output_ds.uri
        )
        else_writer = CompVDataFrameWriter(
            ctx.storage, ctx.tracer, self.output_ds_else.uri
        )

        def _transform(df: pa.Table, selection: pa.Table) -> Tuple[pa.Table, pa.Table]:
            selection = selection.column(0)
            return df.filter(selection), df.filter(pc.invert(selection))

        with selected_writer, else_writer:
            for batch in reader:
                selected_df = CompVDataFrame({}, self.input_ds.system_info)
                else_df = CompVDataFrame({}, self.input_ds.system_info)
                selection, err = fit_pyu(_fit)(batch.data(fit_pyu), expr)
                err = reveal(err)
                if err:
                    raise err
                for pyu in batch.partitions:
                    selected_data, else_data = pyu(_transform)(
                        batch.data(pyu), selection.to(pyu)
                    )
                    selected_df.set_data(selected_data)
                    else_df.set_data(else_data)
                selected_writer.write(selected_df)
                else_writer.write(else_df)

        selected_writer.dump_to(self.output_ds)
        else_writer.dump_to(self.output_ds_else)
