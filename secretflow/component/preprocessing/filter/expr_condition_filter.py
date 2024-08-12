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

import sqlite3

import pandas as pd

from secretflow.component.component import Component, IoType
from secretflow.component.data_utils import DistDataType, dump_table, load_table
from secretflow.data.core import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import PYU, reveal
from secretflow.spec.v1.data_pb2 import IndividualTable, TableSchema, VerticalTable

expr_condition_filter_comp = Component(
    "expr_condition_filter",
    domain="data_filter",
    version="0.0.1",
    desc="""
    Only row-level filtering is supported, column processing is not available;
    the custom expression must comply with SQLite syntax standards
    """,
)

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
    for column in sqlglot.parse_one(sql, dialect="sqlite").find_all(exp.Column):
        columns.append(column.sql().strip('"'))

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

    if in_ds.type == DistDataType.VERTICAL_TABLE:
        meta = VerticalTable()
        in_ds.meta.Unpack(meta)
        schemas = {dr.party: s for s, dr in zip(meta.schemas, in_ds.data_refs)}
        owner = _find_owner(schemas, expr)
    else:
        meta = IndividualTable()
        in_ds.meta.Unpack(meta)
        owner = in_ds.data_refs[0].party

    def _fit(df: pd.DataFrame, expr: str):
        sql = f"SELECT CASE WHEN {expr} THEN TRUE ELSE FALSE END AS hit FROM __table__;"

        try:
            con = sqlite3.connect(":memory:")
            df.to_sql("__table__", con, if_exists="replace", index=False)
            sql_output = pd.read_sql_query(sql, con, dtype={"hit": "bool"})
            filter = pd.Series(sql_output["hit"])
            con.close()
            return filter, None
        except pd.errors.DatabaseError as e:
            return None, ValueError(e.__cause__)
        except Exception as e:
            return None, e

    x = load_table(ctx, in_ds, load_features=True, load_ids=True, load_labels=True)

    owner_pyu = PYU(owner)
    owner_party = x.partitions[owner_pyu]
    filter_series, err_obj = owner_pyu(_fit, num_returns=2)(owner_party.data, expr)
    err = reveal(err_obj)
    if err is not None:
        raise err

    def _transform(df: pd.DataFrame, filter: pd.Series):
        return df[filter], df[~filter]

    out_partitions = {}
    else_partitions = {}
    for pyu, party in x.partitions.items():
        out_data, else_data = pyu(_transform, num_returns=2)(
            party.data, filter_series.to(pyu)
        )
        out_partitions[pyu] = partition(out_data)
        else_partitions[pyu] = partition(else_data)

    out_df = VDataFrame(partitions=out_partitions, aligned=x.aligned)
    out_db = dump_table(ctx, out_df, out_ds, meta, in_ds.system_info)

    else_df = VDataFrame(partitions=else_partitions, aligned=x.aligned)
    else_db = dump_table(ctx, else_df, out_ds_else, meta, in_ds.system_info)

    return {"out_ds": out_db, "out_ds_else": else_db}
