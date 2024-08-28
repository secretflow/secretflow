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

from secretflow.component.component import Component, IoType
from secretflow.component.data_utils import DistDataType, extract_data_infos
from secretflow.component.dataframe import StreamingReader, StreamingWriter
from secretflow.device import PYU, reveal
from secretflow.spec.v1.data_pb2 import TableSchema
from secretflow.utils.consistent_ops import unique_list

expr_condition_filter_comp = Component(
    "expr_condition_filter",
    domain="data_filter",
    version="0.0.1",
    desc="""
    Only row-level filtering is supported, column processing is not available;
    the custom expression must comply with SQLite syntax standards
    """,
)

# Suggest that the user verify the validity of the expression
expr_condition_filter_comp.str_attr(
    name="expr",
    desc="The custom expression must comply with SQLite syntax standards",
    is_list=False,
    is_optional=False,
)

expr_condition_filter_comp.io(
    io_type=IoType.INPUT,
    name="in_ds",
    desc="Input vertical or individual table",
    types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
)

expr_condition_filter_comp.io(
    io_type=IoType.OUTPUT,
    name="out_ds",
    desc="Output table that satisfies the condition",
    types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
)

expr_condition_filter_comp.io(
    io_type=IoType.OUTPUT,
    name="out_ds_else",
    desc="Output table that does not satisfies the condition",
    types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
)


def parse_columns(expr: str) -> list[str]:
    import sqlglot
    import sqlglot.expressions as exp

    sql = f"SELECT * FROM __table__ WHERE {expr}"
    columns = []
    for column in sqlglot.parse_one(sql, dialect="duckdb").find_all(exp.Column):
        columns.append(column.sql().strip('"'))

    columns = unique_list(columns)

    return columns


def _find_owner(schemas: dict[str, TableSchema], expr: str) -> str:  # type: ignore
    columns = parse_columns(expr)
    column_set = set(columns)
    if len(column_set) == 0:
        raise ValueError(f"cannot parse columns from the expr<{expr}>")

    for party, s in schemas.items():
        names = set(list(s.ids) + list(s.labels) + list(s.features))
        if column_set.issubset(names):
            return party
        elif column_set.intersection(names):
            diff = column_set.difference(names)
            raise ValueError(
                f"{diff} is not {party}'s columns. The columns<{columns}> of expr<{expr}> must appears in one party"
            )

    raise ValueError(f"cannot find party by the columns<{columns}> in the expr<{expr}>")


@expr_condition_filter_comp.eval_fn
def expr_condition_filter_comp_eval_fn(*, ctx, expr: str, in_ds, out_ds, out_ds_else):
    expr = expr.strip()
    assert expr != "", f"empty expr"

    columns = parse_columns(expr)

    infos = extract_data_infos(
        in_ds, load_features=True, load_ids=True, load_labels=True, col_selects=columns
    )

    if len(infos) > 1:
        raise AttributeError(
            f"The columns<{columns}> of expr<{expr}> must appears in one party"
        )

    fit_pyu = PYU(list(infos.keys())[0])

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

    reader = StreamingReader.from_distdata(
        ctx,
        in_ds,
        load_features=True,
        load_ids=True,
        load_labels=True,
    )

    selected_writer = StreamingWriter(ctx, out_ds)
    else_writer = StreamingWriter(ctx, out_ds_else)

    def _transform(df: pa.Table, selection: pa.Table) -> Tuple[pa.Table, pa.Table]:
        selection = selection.column(0)
        return df.filter(selection), df.filter(pc.invert(selection))

    with selected_writer, else_writer:
        for batch in reader:
            selected_df = batch.copy()
            else_df = batch.copy()
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

    return {
        "out_ds": selected_writer.to_distdata(),
        "out_ds_else": else_writer.to_distdata(),
    }
