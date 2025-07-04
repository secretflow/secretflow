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

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

import pyarrow as pa
import sqlglot
import sqlglot.expressions as sqlglot_expr
import sqlglot.optimizer
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.canonicalize import canonicalize
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes
from sqlglot.optimizer.eliminate_joins import eliminate_joins
from sqlglot.optimizer.eliminate_subqueries import eliminate_subqueries
from sqlglot.optimizer.merge_subqueries import merge_subqueries
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.optimize_joins import optimize_joins
from sqlglot.optimizer.pushdown_predicates import pushdown_predicates
from sqlglot.optimizer.pushdown_projections import pushdown_projections
from sqlglot.optimizer.qualify_columns import quote_identifiers
from sqlglot.optimizer.simplify import simplify
from sqlglot.optimizer.unnest_subqueries import unnest_subqueries

import secretflow.compute as sc
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
    VTableFieldKind,
    VTableSchema,
    VTableUtils,
    register,
)
from secretflow.utils.consistent_ops import unique_list
from secretflow.utils.errors import InvalidArgumentError

from ..preprocessing import PreprocessingMixin

RULES = (
    pushdown_projections,
    normalize,
    unnest_subqueries,
    pushdown_predicates,
    optimize_joins,
    eliminate_subqueries,
    merge_subqueries,
    eliminate_joins,
    eliminate_ctes,
    quote_identifiers,
    annotate_types,
    canonicalize,
    simplify,
)


@register(domain='preprocessing', version='1.0.0', name="sql_processor")
class SQLProcessor(PreprocessingMixin, Component, IServingExporter):
    '''
    sql processor
    '''

    sql: str = Field.attr(desc="sql for preprocessing, for example SELECT a, b, a+b")

    input_ds: Input = Field.input(
        desc="Input table",
        types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output table",
        types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    )
    output_rule: Output = Field.output(
        desc="Output rule",
        types=[DistDataType.PREPROCESSING_RULE],
    )

    def evaluate(self, ctx: Context):
        input_tbl = VTable.from_distdata(self.input_ds)
        sql_ast = self.parse_sql(self.sql, input_tbl.flatten_schema)

        expressions, trans_tbl = self.do_check(sql_ast, input_tbl)

        rule = self.fit(
            ctx, self.output_rule, trans_tbl, SQLProcessor.do_fit, extras=expressions
        )
        self.transform(ctx, self.output_ds, input_tbl, rule)

    @staticmethod
    def parse_sql(sql: str, schema: VTableSchema) -> sqlglot_expr.Expression:
        sql = sql.rstrip('; \t\n\r')
        if sql.endswith(','):
            raise InvalidArgumentError(
                "the sql cannot have commas.", detail={"sql": sql}
            )
        expressions = sqlglot.parse(sql, dialect="duckdb")
        if len(expressions) != 1:
            raise InvalidArgumentError(
                "the sql can only have one expression.",
                detail={"sql": sql, "expression_size": len(expressions)},
            )
        sql_ast = expressions[0]
        if not isinstance(sql_ast, sqlglot_expr.Select):
            raise InvalidArgumentError(
                "the sql must be a select expression.", detail={"sql": sql}
            )

        table_names = [t.name for t in sql_ast.find_all(sqlglot_expr.Table)]
        if len(table_names) > 1:
            raise InvalidArgumentError(
                "the sql can only have one table.",
                detail={"sql": sql, "table_names": table_names},
            )

        table_name = table_names[0] if len(table_names) == 1 else "*"

        fields = {}
        for f in schema.fields.values():
            fields[f.name] = VTableUtils.to_duckdb_dtype(f.type)
        sql_schema = {"*": {table_name: fields}}

        sql_ast = sqlglot.optimizer.optimize(
            sql_ast, dialect="duckdb", schema=sql_schema, rules=RULES
        )

        for key, clause in sql_ast.args.items():
            if clause is None:
                continue

            # ignore empty joins
            if isinstance(clause, list) and len(clause) == 0:
                continue
            if key not in ["expressions", "from"]:
                raise InvalidArgumentError(
                    "the sql only support simple select clause.",
                    detail={"sql": sql, "key": key, "clause": clause},
                )
        return sql_ast

    @staticmethod
    def do_check(
        sql_ast: sqlglot_expr.Expression, input_tbl: VTable
    ) -> tuple[dict[str, list], VTable]:
        column_names = set(input_tbl.flatten_schema.names)
        column_results = set()
        expressions: dict[str, list] = defaultdict(list)
        all_columns = []
        is_all = False
        for index, expr in enumerate(sql_ast.expressions):
            if isinstance(expr, sqlglot_expr.Star):
                if index != 0:
                    raise InvalidArgumentError("star expr must be the first")
                if is_all:
                    raise InvalidArgumentError("duplicate star expr")
                is_all = True
                for party in input_tbl.parties.keys():
                    expressions[party].append(expr)
                continue

            columns = unique_list(find_columns(expr))

            if len(columns) == 0:
                raise InvalidArgumentError(f"cannot find columns from expr<{expr}>")

            # check col_name and kind
            if isinstance(expr, sqlglot_expr.Alias):
                col_name = expr.alias
                if col_name in column_names and col_name not in columns:
                    columns.append(col_name)
            elif len(columns) != 1:
                raise InvalidArgumentError(
                    "the expr use too many columns, please use alias to specify the name",
                    detail={"columns": columns, "expr": expr},
                )
            else:
                col_name = columns[0]

            if col_name in column_results:
                raise InvalidArgumentError(
                    "duplicate column name",
                    detail={"column_name": col_name, "results": column_results},
                )
            else:
                column_results.add(col_name)

            # check party
            tbl = input_tbl.select(columns)
            if len(tbl.parties) != 1:
                raise InvalidArgumentError(
                    "The columns of expr must appears in one party",
                    detail={"columns": columns, "expr": expr},
                )

            kinds = set(tbl.get_party(0).schema.kinds.values())
            if len(kinds) > 1:
                raise InvalidArgumentError(
                    "all columns should be same kind",
                    detail={"columns": columns, "kinds": kinds},
                )

            party_name = tbl.get_party(0).party
            expressions[party_name].append(expr)
            all_columns.extend(columns)

        if is_all:
            trans_tbl = input_tbl
        else:
            trans_tbl = input_tbl.select(unique_list(all_columns))

        return expressions, trans_tbl

    @staticmethod
    def do_fit(tbl: sc.Table, expressions: list[sqlglot_expr.Expression]) -> sc.Table:
        column_names = {name: idx for idx, name in enumerate(tbl.column_names)}
        new_columns = set()
        all_columns = False
        for expr in expressions:
            if isinstance(expr, sqlglot_expr.Star):
                all_columns = True
                continue

            columns = find_columns(expr)
            assert len(columns) > 0
            first = tbl.schema.field(columns[0])

            if isinstance(expr, sqlglot_expr.Alias):
                col = convert(tbl, expr.this)
                if not first.metadata:
                    kind = VTableFieldKind.FEATURE
                    field = VTableUtils.pa_field(expr.alias, col.dtype, kind)
                else:
                    field = VTableUtils.pa_field_from(expr.alias, col.dtype, first)
            else:
                col = convert(tbl, expr)
                field = first

            col_name = field.name
            new_columns.add(col_name)
            if col_name in column_names:
                tbl = tbl.set_column(column_names[col_name], col_name, col)
            else:
                tbl = tbl.append_column(field, col)

        if not all_columns:
            for col in column_names.keys():
                if col not in new_columns:
                    tbl = tbl.remove_column(col)

        return tbl

    def export(self, ctx: Context, builder: ServingBuilder) -> None:
        self.do_export(ctx, builder, self.input_ds, self.output_rule.data)


def find_columns(expr: sqlglot_expr.Expression) -> list[str]:
    columns_expr = expr.find_all(sqlglot_expr.Column)
    columns = [ce.sql().strip('"') for ce in columns_expr]
    columns = unique_list(columns)
    return columns


@dataclass
class ScalarFunc:
    fn: Callable
    fields: list[str]  # other fields to be parsed
    extend: bool = True  # whether to extend expressions

    def __call__(self, *args, **kwds):
        return self.fn(*args)


_unary_mapping = {
    "bitwisenot": sc.bit_wise_not,
    "not": sc.invert,
    "neg": sc.negate,
    "paren": lambda x: x,
}


def _is_null(l, r: Any | None):
    if r is not None:
        raise TypeError(f"unsupport type<{type(r)},{r}> for is expr")
    return sc.is_null(l)


_binary_mapping = {
    # connector
    "and": sc.and_,
    "or": sc.or_,
    "xor": sc.xor,
    # predicate
    "eq": sc.equal,
    "neq": sc.not_equal,
    "gt": sc.greater,
    "gte": sc.greater_equal,
    "lt": sc.less,
    "lte": sc.less_equal,
    "like": sc.match_like,
    "ilike": lambda l, r: sc.match_like(l, r, ignore_case=True),
    "similarto": sc.match_substring_regex,
    "is": _is_null,
    # bitwise
    "bitwiseand": sc.bit_wise_and,
    "bitwiseor": sc.bit_wise_or,
    "bitwisexor": sc.bit_wise_xor,
    "bitwiseleftshift": sc.shift_left,
    "bitwiserightshift": sc.shift_right,
    # arithmetic
    "add": sc.add,
    "sub": sc.subtract,
    "mul": sc.multiply,
    "div": sc.divide,
    # intdiv return types are floating Current
    "intdiv": lambda l, r: sc.floor(sc.divide(l, r)),
    "mod": lambda l, r: sc.subtract(l, sc.multiply(sc.divide(l, r), r)),
    "pow": sc.power,
    # string
    "dpipe": lambda l, r: sc.binary_join_element_wise(l, r, ""),  # operator ||
}


def _is_in(values, value_set):
    if isinstance(value_set, list):
        value_set = pa.array(value_set)
    return sc.is_in(values, value_set)


def _between(value, low, high):
    return sc.and_(sc.greater_equal(value, low), sc.less_equal(value, high))


_predicate_mapping = {
    "in": ScalarFunc(_is_in, None, extend=False),
    "between": ScalarFunc(_between, ["low", "high"]),
}


def _to_func2(fn1, fn2):
    def wrapper(*args):
        if len(args) == 1:
            return fn1(*args)
        elif len(args) == 2:
            return fn2(*args)
        else:
            raise ValueError("Unsupported number of arguments")

    return wrapper


def _ceil(x, ndigits=0):
    if ndigits == 0:
        return sc.ceil(x)
    factor = 10**ndigits
    x = sc.multiply(x, factor)
    x = sc.ceil(x)
    return sc.divide(x, factor)


def _floor(x, ndigits=0):
    if ndigits == 0:
        return sc.floor(x)
    factor = 10**ndigits
    x = sc.multiply(x, factor)
    x = sc.floor(x)
    return sc.divide(x, factor)


def _trunc(x, ndigits=0):
    if ndigits == 0:
        return sc.trunc(x)
    factor = 10**ndigits
    x = sc.multiply(x, factor)
    x = sc.trunc(x)
    return sc.divide(x, factor)


_scalar_func = {
    "coalesce": sc.coalesce,
    # Categorizations
    "is_nan": sc.is_nan,
    "is_inf": sc.is_inf,
    "is_finite": sc.is_finite,
    "is_null": sc.is_null,
    "is_valid": sc.is_valid,
    # Arithmetic
    "abs": sc.abs,
    "sign": sc.sign,
    "sqrt": sc.sqrt,
    "power": sc.power,
    # Rounding
    "ceil": ScalarFunc(_ceil, ["decimals"]),
    "floor": ScalarFunc(_floor, ["decimals"]),
    "round": ScalarFunc(sc.round, ["decimals"]),
    "trunc": _trunc,
    # Trigonometric
    "acos": sc.acos,
    "asin": sc.asin,
    "atan": sc.atan,
    "atan2": sc.atan2,
    "cos": sc.cos,
    "sin": sc.sin,
    "tan": sc.tan,
    # Logarithmic
    "ln": sc.ln,
    "log10": sc.log10,
    "log1p": sc.log1p,
    "log2": sc.log2,
    "logb": sc.logb,
    "log": lambda *args: (
        sc.logb(args[1], args[0]) if len(args) == 2 else sc.log10(args[0])
    ),
    # String predicate
    "is_alnum": sc.utf8_is_alnum,
    "is_alpha": sc.utf8_is_alpha,
    "is_decimal": sc.utf8_is_decimal,
    "is_digit": sc.utf8_is_digit,
    "is_lower": sc.utf8_is_lower,
    "is_numeric": sc.utf8_is_numeric,
    "is_printable": sc.utf8_is_printable,
    "is_space": sc.utf8_is_space,
    "is_upper": sc.utf8_is_upper,
    "is_title": sc.utf8_is_title,
    # String Transforms
    "repeat": ScalarFunc(sc.binary_repeat, ["times"]),
    "replace": sc.replace_substring,
    "regexp_replace": ScalarFunc(sc.replace_substring_regex, ["replacement"]),
    "capitalize": sc.utf8_capitalize,
    "length": sc.utf8_length,
    "lower": sc.utf8_lower,
    "reverse": sc.utf8_reverse,
    "swapcase": sc.utf8_swapcase,
    "title": sc.utf8_title,
    "upper": sc.utf8_upper,
    # String Padding
    "center": sc.utf8_center,
    "lpad": sc.utf8_lpad,
    "rpad": sc.utf8_rpad,
    # String Trimming
    "trim": _to_func2(sc.utf8_trim_whitespace, sc.utf8_trim),
    "ltrim": _to_func2(sc.utf8_ltrim_whitespace, sc.utf8_ltrim),
    "rtrim": _to_func2(sc.utf8_rtrim_whitespace, sc.utf8_rtrim),
    "concat": lambda *args: sc.binary_join_element_wise(*args, ""),
}

_unsupported_types = tuple(
    [
        sqlglot_expr.Window,
        sqlglot_expr.Parameter,
        sqlglot_expr.SessionParameter,
        sqlglot_expr.Placeholder,
        sqlglot_expr.Bracket,
        # string
        sqlglot_expr.BitString,
        sqlglot_expr.HexString,
        sqlglot_expr.ByteString,
        sqlglot_expr.RawString,
        sqlglot_expr.UnicodeString,
        # Predicate
        sqlglot_expr.SubqueryPredicate,
        sqlglot_expr.JSONArrayContains,
        sqlglot_expr.ILikeAny,
        sqlglot_expr.LikeAny,
        sqlglot_expr.Glob,
        sqlglot_expr.NullSafeEQ,
        sqlglot_expr.NullSafeNEQ,
        # binary
        sqlglot_expr.PropertyEQ,
        sqlglot_expr.Overlaps,
        sqlglot_expr.Dot,
        sqlglot_expr.Distance,
        sqlglot_expr.Escape,
        sqlglot_expr.Kwarg,
        sqlglot_expr.Operator,
        sqlglot_expr.Slice,
        sqlglot_expr.ArrayContains,
        sqlglot_expr.ArrayContainsAll,
        sqlglot_expr.ArrayOverlaps,
        sqlglot_expr.Collate,
        sqlglot_expr.JSONExtract,
        sqlglot_expr.JSONExtractScalar,
        sqlglot_expr.JSONBExtract,
        sqlglot_expr.JSONBExtractScalar,
        sqlglot_expr.JSONBContains,
        sqlglot_expr.JSONArrayContains,
        sqlglot_expr.RegexpLike,
        sqlglot_expr.RegexpILike,
        sqlglot_expr.Corr,
        sqlglot_expr.CovarSamp,
        sqlglot_expr.CovarPop,
        # Func
        sqlglot_expr.AggFunc,
        sqlglot_expr.AnonymousAggFunc,
    ]
)


def _to_type(dtype: sqlglot_expr.DataType) -> pa.DataType:
    if dtype.is_type(sqlglot_expr.DataType.Type.BOOLEAN):
        return pa.bool_()
    elif dtype.is_type(sqlglot_expr.DataType.Type.FLOAT):
        return pa.float32()
    elif dtype.is_type(sqlglot_expr.DataType.Type.DOUBLE):
        return pa.float64()
    elif dtype.is_type(*sqlglot_expr.DataType.TEXT_TYPES):
        return pa.string()
    elif dtype.is_type(*sqlglot_expr.DataType.SIGNED_INTEGER_TYPES):
        return pa.int64()
    elif dtype.is_type(*sqlglot_expr.DataType.UNSIGNED_INTEGER_TYPES):
        return pa.uint64()
    else:
        raise ValueError(f"unsupported data type<{dtype}>")


def _invoke(
    fn: Callable,
    tbl: sc.Table,
    expr: sqlglot_expr.Expression,
    extensions: list[str] = None,
):
    fields, extend = None, True
    if isinstance(fn, ScalarFunc):
        fields = fn.fields
        extend = fn.extend

    args = []

    if "this" in expr.args:
        args.append(convert(tbl, expr.this))

    if "expression" in expr.args:
        args.append(convert(tbl, expr.expression))
    elif "expressions" in expr.args:
        expressions = [convert(tbl, n) for n in expr.expressions]
        if extend:
            args.extend(expressions)
        else:
            args.append(expressions)

    if fields:
        for key in fn.fields:
            if key not in expr.args:
                continue
            v = expr.args[key]
            if isinstance(v, list):
                args.append([convert(tbl, n) for n in v])
            else:
                args.append(convert(tbl, v))

    return _invoke_func(fn, args, expr)


def _invoke_func(fn: Callable, args: list, expr: sqlglot_expr.Expression):
    try:
        return fn(*args)
    except Exception as e:
        raise ValueError(f"invoke fail, error={e}, expr={expr}")


# pyarrow.compute: https://arrow.apache.org/docs/python/api/compute.html
def convert(tbl: sc.Table, expr: sqlglot_expr.Expression):
    if not isinstance(expr, sqlglot_expr.Condition):
        raise TypeError(f"unsupported type<{type(expr)}> {expr}")
    if isinstance(expr, _unsupported_types):
        raise TypeError(f"unsupported condition type<{type(expr)}> {expr.sql()}")

    if isinstance(expr, sqlglot_expr.Column):
        return tbl.column(expr.name)
    elif isinstance(expr, sqlglot_expr.Literal):
        v = expr.to_py()
        return v if not isinstance(v, Decimal) else float(v)
    elif isinstance(expr, (sqlglot_expr.Null, sqlglot_expr.Boolean)):
        return expr.to_py()
    elif isinstance(expr, sqlglot_expr.Unary):
        assert expr.key in _unary_mapping, f"unknown unary operator<{expr.key}>"
        return _invoke(_unary_mapping[expr.key], tbl, expr)
    elif isinstance(expr, sqlglot_expr.Binary):
        if expr.key not in _binary_mapping:
            raise TypeError(f"unsupported binary operator<{expr.key},{expr.sql()}>")
        return _invoke(_binary_mapping[expr.key], tbl, expr)
    elif isinstance(expr, sqlglot_expr.Predicate):
        if expr.key not in _predicate_mapping:
            raise TypeError(f"unsupported predicate type<{type(expr)}> {expr.sql()}")
        return _invoke(_predicate_mapping[expr.key], tbl, expr)

    if not isinstance(expr, sqlglot_expr.Func):
        raise TypeError(f"unknown type<{type(expr)}> {expr.sql()}")

    if isinstance(expr, sqlglot_expr.Case):
        conds = []
        cases = []
        for if_expr in expr.args["ifs"]:
            assert isinstance(
                if_expr, sqlglot_expr.If
            ), f"unknown type<{type(if_expr)}> {if_expr}"
            conds.append(convert(tbl, if_expr.this))
            cases.append(convert(tbl, if_expr.args["true"]))
        if 'default' in expr.args:
            cases.append(convert(tbl, expr.args['default']))
        conds = sc.make_struct(*conds)
        return sc.case_when(conds, *cases)
    elif isinstance(expr, sqlglot_expr.Cast):
        value = convert(tbl, expr.this)
        dtype = _to_type(expr.to)
        return _invoke_func(sc.cast, [value, dtype], expr)
    elif isinstance(expr, sqlglot_expr.Anonymous):
        func_name = expr.name.lower()
        if func_name not in _scalar_func:
            raise ValueError(f"unsupported anonymous func<{expr.name}> {expr.args}")
        args = [convert(tbl, n) for n in expr.expressions if expr.expressions]
        fn = _scalar_func[func_name]
        return _invoke_func(fn, args, expr)
    else:
        sql_name = ""
        for name in expr.sql_names():
            name = name.lower()
            if name in _scalar_func:
                sql_name = name
                break
        if sql_name == "":
            raise ValueError(f"unsupported func<{expr.key}> {expr.sql()}")

        scalar_fn = _scalar_func[sql_name]

        return _invoke(scalar_fn, tbl, expr)
