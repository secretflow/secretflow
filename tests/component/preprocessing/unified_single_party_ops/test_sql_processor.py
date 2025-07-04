# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

import pandas as pd
import pyarrow as pa
import pytest

import secretflow.compute as sc
from secretflow.component.core import (
    VTable,
    VTableParty,
    VTableUtils,
    assert_almost_equal,
    build_node_eval_param,
    comp_eval,
    make_storage,
    read_orc,
)
from secretflow.component.preprocessing.unified_single_party_ops.sql_processor import (
    SQLProcessor,
)


@pytest.mark.mpc
def test_sql_processor(sf_production_setup_comp):
    work_dir = "test_sql_processor"
    out_ds = f"{work_dir}/output_ds"
    out_rule = f"{work_dir}/output_rule"

    storage_config, sf_cluster_config = sf_production_setup_comp
    self_party = sf_cluster_config.private_config.self_party
    storage = make_storage(storage_config)

    datas = {
        "alice": {"A1": [1, 2, 3], "A2": [0.1, 0.2, 0.3]},
        "bob": {"B1": ["b1", "b2", "b3"], "B2": [0.11, 0.22, 0.33]},
    }
    if self_party in datas:
        path = f"{work_dir}/{self_party}"
        with storage.get_writer(path) as w:
            pd.DataFrame(datas[self_party]).to_csv(w, index=False)

    param = build_node_eval_param(
        domain="preprocessing",
        name="sql_processor",
        version="1.0.0",
        attrs={"sql": "SELECT *, A1 + A2 as A3, B2 + 0.1 as B2"},
        inputs=[
            VTable(
                name="sql_processor",
                parties=[
                    VTableParty.from_dict(
                        uri=f"{work_dir}/alice",
                        party="alice",
                        format="csv",
                        features={"A1": "int", "A2": "float"},
                    ),
                    VTableParty.from_dict(
                        uri=f"{work_dir}/bob",
                        party="bob",
                        format="csv",
                        ids={"B1": "str"},
                        features={"B2": "float"},
                    ),
                ],
            )
        ],
        output_uris=[out_ds, out_rule],
    )

    res = comp_eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2
    excepted = {
        "alice": {"A1": [1, 2, 3], "A2": [0.1, 0.2, 0.3], "A3": [1.1, 2.2, 3.3]},
        "bob": {"B1": ["b1", "b2", "b3"], "B2": [0.21, 0.32, 0.43]},
    }
    if self_party in datas:
        output_tbl = read_orc(storage.get_reader(out_ds))
        logging.warning(f"{output_tbl}")
        excepted_tbl = pa.Table.from_pydict(excepted[self_party])
        assert_almost_equal(output_tbl, excepted_tbl, ignore_order=True)


def test_sql_processor_run_sql():
    def to_pa_table(datas: dict, types: dict):
        fields = []
        for name, value in datas.items():
            if name in types:
                field_type = types[name]
            else:
                field_type = pa.array(value).type
            fields.append(pa.field(name, field_type))
        return pa.Table.from_pydict(datas, schema=pa.schema(fields))

    def run(
        name: str, sql: str, input_tbl: pa.Table | dict, excepted_tbl: pa.Table | dict
    ):
        if isinstance(input_tbl, dict):
            input_tbl = pa.Table.from_pydict(input_tbl)
        if isinstance(excepted_tbl, dict):
            excepted_tbl = pa.Table.from_pydict(excepted_tbl)

        try:
            ast = SQLProcessor.parse_sql(
                sql, VTableUtils.from_arrow_schema(input_tbl.schema, check_kind=False)
            )
        except Exception as e:
            logging.warning(f"parse sql fail, name={name}, sql={sql}")
            raise e

        sc_input_tbl = sc.Table.from_schema(input_tbl.schema)
        sc_output_tbl = SQLProcessor.do_fit(sc_input_tbl, ast.expressions)
        runner = sc_output_tbl.dump_runner()
        output_tbl = runner.run(input_tbl)
        assert_almost_equal(output_tbl, excepted_tbl)

    test_cases = [
        {
            "name": "arithmetic",
            "sql": "SELECT a + 1 as a1, a -1 as a2, a*2 as a3, a/2 as a4, a//2 as a5, a%2 as a6, b/2 as b1, -b as b2, not c as c1 FROM my_table;",
            "data": {"a": [3], "b": [3.2], "c": [True]},
            "excepted": {
                "a1": [4],
                "a2": [2],
                "a3": [6],
                "a4": [1],
                "a5": [1.0],  # type mismatch
                "a6": [1],
                "b1": [1.6],
                "b2": [-3.2],
                "c1": [False],
            },
        },
        {
            "name": "bitwise",
            "sql": "SELECT a1 & 15, a2 |3, a3 << 4, a4 >> 2, a5 ^ 4, ~a6 ",
            "data": {
                "a1": [91],
                "a2": [32],
                "a3": [1],
                "a4": [8],
                "a5": [3],
                "a6": [15],
            },
            "excepted": {
                "a1": [11],
                "a2": [35],
                "a3": [16],
                "a4": [2],
                "a5": [7],
                "a6": [-16],
            },
        },
        {
            "name": "paren",
            "sql": "SELECT (a + b) * c-d as r",
            "data": {"a": [1], "b": [2], "c": [3], "d": [4]},
            "excepted": {"r": [5]},
        },
        {
            "name": "case_when",
            "sql": "SELECT CASE WHEN a in (1,2) THEN 'one' WHEN b BETWEEN 4 AND 6 THEN 'five' ELSE 'other' END as c",
            "data": {'a': [1, 0, 2], 'b': [3, 5, 7]},
            "excepted": {'c': ['one', 'five', 'one']},
        },
        {
            "name": "null",
            "sql": "SELECT CASE WHEN a is null THEN 0 WHEN b is NOT null THEN 1 ELSE 2 END as c",
            "data": {"a": [None, 1, 2], "b": [3, 4, None]},
            "excepted": {"c": [0, 1, 2]},
        },
        {
            "name": "star",
            "sql": "SELECT *, A1 + A2 as A3, B2 + 0.1 AS B3",
            "data": {
                "A1": [1, 2, 3],
                "A2": [0.1, 0.2, 0.3],
                "B1": ["b1", "b2", "b3"],
                "B2": [0.11, 0.22, 0.33],
            },
            "excepted": {
                "A1": [1, 2, 3],
                "A2": [0.1, 0.2, 0.3],
                "B1": ["b1", "b2", "b3"],
                "B2": [0.11, 0.22, 0.33],
                "A3": [1.1, 2.2, 3.3],
                "B3": [0.21, 0.32, 0.43],
            },
        },
        {
            "name": "cast",
            "sql": "SELECT cast(a as VARCHAR) as a1, cast(a as DOUBLE) as a2, cast(b as INT) as b1, cast(b as TEXT) as b2, cast(c as INT) as c1, cast(c as DOUBLE) as c2",
            "data": {"a": [1], "b": [1.0], "c": ["1"]},
            "excepted": {
                "a1": ["1"],
                "a2": [1.0],
                "b1": [1],
                "b2": ["1"],
                "c1": [1],
                "c2": [1.0],
            },
        },
        {
            "name": "coalesce",
            "sql": "SELECT COALESCE(a, 0), COALESCE(b, 'NULL')",
            "data": {"a": [1, None], "b": [None, "xx"]},
            "excepted": {"a": [1, 0], "b": ["NULL", "xx"]},
        },
        {
            # https://duckdb.org/docs/sql/functions/numeric
            "name": "numeric func",
            "sql": "SELECT log(a) as a1, log10(a) as a2, log(10, a) as a3, ln(a) as a4, cos(a) as a5",
            "data": {"a": [100]},
            "excepted": {
                "a1": [2.0],
                "a2": [2.0],
                "a3": [2.0],
                "a4": [4.605170],
                "a5": [0.862318],
            },
        },
        {
            "name": "string func",
            "sql": "SELECT title(a) as a1, repeat(a,2) as a2, length(a) as a3, \
                replace(a, 'he', 'ha') as a4, regexp_replace(a, '[lo]', '-') as a5, \
                ltrim(b) as b1, rtrim(b, '$') as b2, reverse(b) as b3, rpad(b, 5, '-') as b4,\
                concat(a,b) as c1",
            "data": {"a": ["hello"], "b": [" h$"]},
            "excepted": to_pa_table(
                {
                    "a1": ["Hello"],
                    "a2": ["hellohello"],
                    "a3": [5],
                    "a4": ["hallo"],
                    "a5": ["he---"],
                    "b1": ["h$"],
                    "b2": [" h"],
                    "b3": ["$h "],
                    "b4": [" h$--"],
                    "c1": ["hello h$"],
                },
                {"a3": pa.int32()},
            ),
        },
        {
            "name": "concat",
            "sql": "SELECT COALESCE(a, '') || COALESCE(b, '') as c",
            "data": {"a": ["hello", None, "prefix"], "b": ["world", "suffix", None]},
            "excepted": {"c": ["helloworld", "suffix", "prefix"]},
        },
        {
            "name": "replace column c and annotation",
            "sql": "/*it is an annotation.*/SELECT a + b as c",
            "data": {"a": [1], "b": [2], "c": [0.1]},
            "excepted": {"c": [3]},
        },
        {
            "name": "rounding",
            "sql": "SELECT CEIL(a) as a1, FLOOR(a) as a2, ROUND(a) as a3, TRUNC(a) as a4, \
                CEIL(a, 2) as b1, FLOOR(a, 2) as b2, ROUND(a, 2) as b3, TRUNC(a, 2) as b4",
            "data": {"a": [1.2345, 2.3456, -3.4567, -4.5678]},
            "excepted": {
                "a1": [2.0, 3, -3, -4],
                "a2": [1.0, 2, -4, -5],
                "a3": [1.0, 2, -3, -5],
                "a4": [1.0, 2, -3, -4],
                "b1": [1.24, 2.35, -3.45, -4.56],
                "b2": [1.23, 2.34, -3.46, -4.57],
                "b3": [1.23, 2.35, -3.46, -4.57],
                "b4": [1.23, 2.34, -3.45, -4.56],
            },
        },
    ]

    for c in test_cases:
        sqls = c["sqls"] if "sqls" in c else [c["sql"]]
        if isinstance(sqls, str):
            sqls = [sqls]
        for sql in sqls:
            run(c["name"], sql, c["data"], c["excepted"])


def test_sql_processor_error():
    test_cases = [
        {
            "name": "zero column",
            "sql": "SELECT a1, 1",
        },
        {
            "name": "different kind",
            "sql": "SELECT a1 + y as c",
        },
        {
            "name": "no column name",
            "sql": "SELECT a1 + a2",
        },
        {
            "name": "duplicate column name, a1",
            "sql": "SELECT a1, a1 + a2 as a1",
        },
        {
            "name": "should be one party",
            "sql": "SELECT a1 + b1 as c",
        },
        {
            "name": "no distinct",
            "sql": "SELECT distinct a1",
        },
        {
            "name": "no where",
            "sql": "SELECT a1 from table where a1 > 1",
        },
        {
            "name": "star must be first",
            "sql": "SELECT a1, *, a2",
        },
        {
            "name": "type mismatch",
            "sql": "SELECT a1 + c as col1",
        },
        {
            "name": "multi sql expressions",
            "sql": "SELECT a1; SELECT a2",
        },
        {
            "name": "extra commas",
            "sql": "SELECT *,",
        },
    ]

    input_tbl = VTable(
        name="",
        parties=[
            VTableParty.from_dict(
                party="alice",
                features={"a1": "int", "a2": "int", "c": "str"},
                labels={"y": "int"},
            ),
            VTableParty.from_dict(party="bob", features={"b1": "int"}),
        ],
    )

    def run(name: str, sql: str, input_tbl: VTable):
        with pytest.raises(Exception) as exc_info:
            ast = SQLProcessor.parse_sql(sql, input_tbl.flatten_schema)
            expressions, tran_tbl = SQLProcessor.do_check(ast, input_tbl)
            for p in tran_tbl.parties.values():
                schema = VTableUtils.to_arrow_schema(p.schema)
                sc_tbl = sc.Table.from_schema(schema)
                SQLProcessor.do_fit(sc_tbl, expressions[p.party])
            logging.info(f"expect exception {name}, {sql}, {exc_info}")

    for tc in test_cases:
        run(tc["name"], tc["sql"], input_tbl)
