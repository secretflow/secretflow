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

from typing import Dict, List, Union

import pyarrow as pa
from pyarrow import compute as pc

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
    float_almost_equal,
    register,
)

from ..preprocessing import PreprocessingMixin


def apply_fillna_rule_on_table(table: sc.Table, rules: Dict) -> sc.Table:
    nan_is_null = rules["nan_is_null"]
    rules = rules["fill_rules"]
    for col_name, rule in rules.items():
        outliers = rule["outlier_values"]
        fill_value = rule["fill_value"]
        col = table.column(col_name)
        dtype = col.dtype

        if isinstance(fill_value, float):
            assert pa.types.is_floating(dtype)
            equal_fn = float_almost_equal
        elif isinstance(fill_value, bool):
            assert pa.types.is_boolean(dtype)
            equal_fn = sc.equal
        elif isinstance(fill_value, int):
            assert pa.types.is_integer(
                dtype
            ), f"col_name {col_name}, fill_value {fill_value}, dtype {dtype}"
            equal_fn = sc.equal
        elif isinstance(fill_value, str):
            assert pa.types.is_string(dtype)
            equal_fn = sc.equal
        else:
            raise RuntimeError(f"unknown fill_value type {type(fill_value)}")

        col = sc.if_else(sc.is_null(col, nan_is_null=nan_is_null), fill_value, col)

        cond = None
        for o in outliers:
            if cond:
                cond = sc.or_(cond, equal_fn(col, o))
            else:
                cond = equal_fn(col, o)

        if cond:
            col = sc.if_else(cond, fill_value, col)
        table = table.set_column(table.column_names.index(col_name), col_name, col)
    return table


def most_frequent(col: pa.ChunkedArray):
    value_counts = pc.value_counts(col)
    sorted_indices = pc.select_k_unstable(
        value_counts.field(1), k=1, sort_keys=((0, "descending"),)
    )
    return value_counts[sorted_indices[0].as_py()][0].as_py()


def float_not_equal(c, o, epsilon: float = 1e-07):
    return pc.greater(pc.abs(pc.subtract(c, o)), epsilon)


def clear_col(
    col: pa.ChunkedArray, outliers: List[Union[int, float, str]], nan_is_null: bool
):
    dtype = col.type
    col = pc.drop_null(col)
    if pa.types.is_string(dtype) or pa.types.is_integer(dtype):
        not_equal_fn = pc.not_equal
    elif pa.types.is_floating(dtype):
        not_equal_fn = float_not_equal
    else:
        return col

    selection = None
    if pa.types.is_floating(dtype) and nan_is_null:
        selection = pc.invert(pc.is_nan(col))
    for o in outliers:
        if selection is None:
            selection = not_equal_fn(col, o)
        else:
            selection = pc.and_(not_equal_fn(col, o), selection)

    if selection:
        col = pc.filter(col, selection)
    return col


def fit_col(
    col_name: str,
    col: pa.ChunkedArray,
    outliers: List[Union[int, float, str]],
    fill_strategy: str,
    fill_value: Union[int, float, str],
    nan_is_null: bool,
):
    dtype = col.type
    col = clear_col(col, outliers, nan_is_null)
    if fill_strategy != "constant" and col.length() == 0:
        raise AttributeError(
            f"Column {col_name} contains only null values and outlie values, "
            "no available most frequent value."
        )

    if fill_strategy == "constant":
        pass
    elif fill_strategy == "most_frequent":
        fill_value = most_frequent(col)
    elif (
        not pa.types.is_string(dtype)
        and not pa.types.is_boolean(dtype)
        and fill_strategy == "mean"
    ):
        fill_value = pc.mean(col).as_py()
    elif (
        not pa.types.is_string(dtype)
        and not pa.types.is_boolean(dtype)
        and fill_strategy == "median"
    ):
        fill_value = pc.approximate_median(col).as_py()
    else:
        raise AttributeError(
            f"unsupported fill_strategy {fill_strategy} for dtype {dtype} col"
        )

    if pa.types.is_integer(dtype):
        fill_value = round(fill_value)

    rule = {"outlier_values": outliers, "fill_value": fill_value}
    return rule


@register(domain="preprocessing", version="1.0.0", name="fillna")
class FillNA(PreprocessingMixin, Component):
    '''
    Fill null/nan or other specificed outliers in dataset
    '''

    nan_is_null: bool = Field.attr(
        desc="Whether floating-point NaN values are considered null, take effect with float columns",
        default=True,
    )
    float_outliers: list[float] = Field.attr(
        desc="These outlier value are considered null, take effect with float columns",
        default=[],
    )
    int_outliers: list[int] = Field.attr(
        desc="These outlier value are considered null, take effect with int columns",
        default=[],
    )
    str_outliers: list[str] = Field.attr(
        desc="These outlier value are considered null, take effect with str columns",
        default=[],
    )
    str_fill_strategy: str = Field.attr(
        desc="""Replacement strategy for str column.
        If "most_frequent", then replace missing using the most frequent value along each column.

        If "constant", then replace missing values with fill_value_str.
        """,
        default="constant",
        choices=["constant", "most_frequent"],
    )
    fill_value_str: str = Field.attr(
        desc="For str type data. If method is 'constant' use this value for filling null.",
        default="",
    )
    int_fill_strategy: str = Field.attr(
        desc="""Replacement strategy for int column.
        If "mean", then replace missing values using the mean along each column.

        If "median", then replace missing values using the median along each column

        If "most_frequent", then replace missing using the most frequent value along each column.

        If "constant", then replace missing values with fill_value_int.
        """,
        default="constant",
        choices=["mean", "median", "most_frequent", "constant"],
    )
    fill_value_int: int = Field.attr(
        desc="For int type data. If method is 'constant' use this value for filling null.",
        default=0,
    )
    float_fill_strategy: str = Field.attr(
        desc="""Replacement strategy for float column.
        If "mean", then replace missing values using the mean along each column.

        If "median", then replace missing values using the median along each column

        If "most_frequent", then replace missing using the most frequent value along each column.

        If "constant", then replace missing values with fill_value_float.
        """,
        default="constant",
        choices=["mean", "median", "most_frequent", "constant"],
    )
    fill_value_float: float = Field.attr(
        desc="For float type data. If method is 'constant' use this value for filling null.",
        default=0.0,
    )
    bool_fill_strategy: str = Field.attr(
        desc="""Replacement strategy for bool column.
        If "most_frequent", then replace missing using the most frequent value along each column.

        If "constant", then replace missing values with fill_value_bool.
        """,
        default="constant",
        choices=["constant", "most_frequent"],
    )
    fill_value_bool: bool = Field.attr(
        desc="For bool type data. If method is 'constant' use this value for filling null.",
        default=False,
    )
    fill_na_features: list[str] = Field.table_column_attr(
        "input_ds",
        desc="Features to fill.",
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
        desc="fill value rule",
        types=[DistDataType.PREPROCESSING_RULE],
    )

    def evaluate(self, ctx: Context):
        def _fit(data: pa.Table):
            fill_rules = {}
            for n in data.column_names:
                col = data.column(n)
                dtype = col.type
                if pa.types.is_string(dtype):
                    fill_rules[n] = fit_col(
                        n,
                        col,
                        self.str_outliers,
                        self.str_fill_strategy,
                        self.fill_value_str,
                        False,
                    )
                elif pa.types.is_integer(dtype):
                    fill_rules[n] = fit_col(
                        n,
                        col,
                        self.int_outliers,
                        self.int_fill_strategy,
                        self.fill_value_int,
                        False,
                    )
                elif pa.types.is_floating(dtype):
                    fill_rules[n] = fit_col(
                        n,
                        col,
                        self.float_outliers,
                        self.float_fill_strategy,
                        self.fill_value_float,
                        self.nan_is_null,
                    )
                elif pa.types.is_boolean(dtype):
                    fill_rules[n] = fit_col(
                        n, col, [], self.bool_fill_strategy, self.fill_value_bool, False
                    )
                else:
                    raise AttributeError(f"unsupported column dtype {dtype} ")

            return {"nan_is_null": self.nan_is_null, "fill_rules": fill_rules}

        def _fillna_fit_transform(trans_data: sc.Table) -> sc.Table:
            fillna_rules = _fit(trans_data.to_table())
            trans_data = apply_fillna_rule_on_table(trans_data, fillna_rules)
            return trans_data

        in_tbl = VTable.from_distdata(self.input_ds)
        tran_tbl = in_tbl.select(self.fill_na_features)
        self.fit_transform(
            ctx,
            self.output_rule,
            self.output_ds,
            in_tbl,
            tran_tbl,
            _fillna_fit_transform,
        )

    def export(self, ctx: Context, builder: ServingBuilder) -> None:
        self.do_export(ctx, builder, self.input_ds, self.output_rule.data)
