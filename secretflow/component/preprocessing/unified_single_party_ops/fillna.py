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
import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import compute as pc
from sklearn.impute import SimpleImputer

import secretflow.compute as sc
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import SUPPORTED_VTABLE_DATA_TYPE, DistDataType
from secretflow.component.preprocessing.core.table_utils import (
    float_almost_equal,
    v_preprocessing_transform,
)

fillna = Component(
    "fillna",
    domain="preprocessing",
    version="1.0.0",
    desc="Fill null/nan or other specificed outliers in dataset",
)

# what is null
fillna.bool_attr(
    name="nan_is_null",
    desc="Whether floating-point NaN values are considered null, take effect with float columns",
    is_list=False,
    is_optional=True,
    default_value=True,
)

fillna.float_attr(
    name="float_outliers",
    desc="These outlier value are considered null, take effect with float columns",
    is_list=True,
    is_optional=True,
    default_value=[],
)

fillna.int_attr(
    name="int_outliers",
    desc="These outlier value are considered null, take effect with int columns",
    is_list=True,
    is_optional=True,
    default_value=[],
)

fillna.str_attr(
    name="str_outliers",
    desc="These outlier value are considered null, take effect with str columns",
    is_list=True,
    is_optional=True,
    default_value=[],
)

# how to fill
fillna.str_attr(
    name="str_fill_strategy",
    desc="""Replacement strategy for str column.
    If "most_frequent", then replace missing using the most frequent value along each column.

    If "constant", then replace missing values with fill_value_str.
    """,
    is_list=False,
    is_optional=True,
    default_value="constant",
    allowed_values=["constant", "most_frequent"],
)

fillna.str_attr(
    name="fill_value_str",
    desc="For str type data. If method is 'constant' use this value for filling null.",
    is_list=False,
    is_optional=True,
    default_value="",
)

fillna.str_attr(
    name="int_fill_strategy",
    desc="""Replacement strategy for int column.
    If "mean", then replace missing values using the mean along each column.

    If "median", then replace missing values using the median along each column

    If "most_frequent", then replace missing using the most frequent value along each column.

    If "constant", then replace missing values with fill_value_int.
    """,
    is_list=False,
    is_optional=True,
    default_value="constant",
    allowed_values=["mean", "median", "most_frequent", "constant"],
)

fillna.int_attr(
    name="fill_value_int",
    desc="For int type data. If method is 'constant' use this value for filling null.",
    is_list=False,
    is_optional=True,
    default_value=0,
)

fillna.str_attr(
    name="float_fill_strategy",
    desc="""Replacement strategy for float column.
    If "mean", then replace missing values using the mean along each column.

    If "median", then replace missing values using the median along each column

    If "most_frequent", then replace missing using the most frequent value along each column.

    If "constant", then replace missing values with fill_value_float.
    """,
    is_list=False,
    is_optional=True,
    default_value="constant",
    allowed_values=["mean", "median", "most_frequent", "constant"],
)

fillna.float_attr(
    name="fill_value_float",
    desc="For float type data. If method is 'constant' use this value for filling null.",
    is_list=False,
    is_optional=True,
    default_value=0.0,
)

fillna.str_attr(
    name="bool_fill_strategy",
    desc="""Replacement strategy for bool column.
    If "most_frequent", then replace missing using the most frequent value along each column.

    If "constant", then replace missing values with fill_value_bool.
    """,
    is_list=False,
    is_optional=True,
    default_value="constant",
    allowed_values=["constant", "most_frequent"],
)


fillna.bool_attr(
    name="fill_value_bool",
    desc="For bool type data. If method is 'constant' use this value for filling null.",
    is_list=False,
    is_optional=True,
    default_value=False,
)


fillna.io(
    io_type=IoType.INPUT,
    name="input_dataset",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[TableColParam(name="fill_na_features", desc="Features to fill.")],
)


fillna.io(
    io_type=IoType.OUTPUT,
    name="out_ds",
    desc="Output vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)


fillna.io(
    io_type=IoType.OUTPUT,
    name="out_rules",
    desc="fill value rule",
    types=[DistDataType.PREPROCESSING_RULE],
    col_params=None,
)


# current version 2.0
MODEL_MAX_MAJOR_VERSION = 2
MODEL_MAX_MINOR_VERSION = 0


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


@fillna.eval_fn
def fillna_eval_fn(
    *,
    ctx,
    nan_is_null,
    float_outliers,
    int_outliers,
    str_outliers,
    str_fill_strategy,
    fill_value_str,
    int_fill_strategy,
    fill_value_int,
    float_fill_strategy,
    fill_value_float,
    bool_fill_strategy,
    fill_value_bool,
    input_dataset,
    input_dataset_fill_na_features,
    out_ds,
    out_rules,
):
    def fit(data: pa.Table):
        fill_rules = {}
        for n in data.column_names:
            col = data.column(n)
            dtype = col.type
            if pa.types.is_string(dtype):
                fill_rules[n] = fit_col(
                    n, col, str_outliers, str_fill_strategy, fill_value_str, False
                )
            elif pa.types.is_integer(dtype):
                fill_rules[n] = fit_col(
                    n, col, int_outliers, int_fill_strategy, fill_value_int, False
                )
            elif pa.types.is_floating(dtype):
                fill_rules[n] = fit_col(
                    n,
                    col,
                    float_outliers,
                    float_fill_strategy,
                    fill_value_float,
                    nan_is_null,
                )
            elif pa.types.is_boolean(dtype):
                fill_rules[n] = fit_col(
                    n, col, [], bool_fill_strategy, fill_value_bool, False
                )
            else:
                raise AttributeError(f"unsupported column dtype {dtype} ")

        return {"nan_is_null": nan_is_null, "fill_rules": fill_rules}

    def fillna_fit_transform(trans_data: pa.Table):
        fillna_rules = fit(trans_data)
        trans_data = apply_fillna_rule_on_table(
            sc.Table.from_pyarrow(trans_data), fillna_rules
        )
        return trans_data, [], fillna_rules

    with ctx.tracer.trace_running():
        (output_dd, model_dd, _) = v_preprocessing_transform(
            ctx,
            input_dataset,
            input_dataset_fill_na_features,
            fillna_fit_transform,
            out_ds,
            out_rules,
            "fillna",
            assert_one_party=False,
        )
    return {"out_ds": output_dd, "out_rules": model_dd}
