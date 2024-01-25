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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.∏
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import secretflow.compute as sc
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import SUPPORTED_VTABLE_DATA_TYPE, DistDataType
from secretflow.component.preprocessing.core.table_utils import (
    v_preprocessing_transform,
)

fillna = Component(
    "fillna",
    domain="preprocessing",
    version="0.0.1",
    desc="fillna",
)

# fill want to do mean or sum fill use stats first and then value fill.
# see pyarrow.compute.fill_null*
SUPPORTED_FILL_NA_METHOD = ["mean", "median", "most_frequent", "constant"]

fillna.str_attr(
    name="strategy",
    desc="""The imputation strategy.
    If "mean", then replace missing values using the mean along each column. Can only be used with numeric data.

    If "median", then replace missing values using the median along each column. Can only be used with numeric data.

    If "most_frequent", then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.

    If "constant", then replace missing values with fill_value. Can be used with strings or numeric data.
    """,
    is_list=False,
    is_optional=True,
    default_value="constant",
    allowed_values=SUPPORTED_FILL_NA_METHOD,
)
NA_SUPPORTED_TYPE_DICT = {
    "general_na": str,
    "str": str,
    "int": int,
    "float": float,
}
fillna.str_attr(
    name="missing_value_type",
    desc="type of missing value. general_na type indicates that only np.nan, None or pandas.NA will be treated as missing values. When the type is not general_na, the type casted missing_value_type(missing_value) will also be treated as missing value as well, \
    in addition to general_na values.",
    is_list=False,
    is_optional=True,
    default_value="general_na",
    allowed_values=list(NA_SUPPORTED_TYPE_DICT.keys()),
)

fillna.str_attr(
    name="missing_value",
    desc="Which value should be treat as missing_value? If missing value type is 'general_na', \
    this field will be ignored, and any np.nan, pd.NA, etc value will be treated as missing value. \
    Otherwise, the type casted missing_value_type(missing_value) will also be treated as missing value as well, \
    in addition to general_na values. \
    In case the cast is not successful, general_na will be used instead. \
    default value is 'custom_missing_value'.",
    is_list=False,
    is_optional=True,
    default_value="custom_missing_value",
)


fillna.int_attr(
    name="fill_value_int",
    desc="For int type data. If method is 'constant' use this value for filling null.",
    is_list=False,
    is_optional=True,
    default_value=0,
)

fillna.float_attr(
    name="fill_value_float",
    desc="For float type data. If method is 'constant' use this value for filling null.",
    is_list=False,
    is_optional=True,
    default_value=0.0,
)
fillna.str_attr(
    name="fill_value_str",
    desc="For str type data. If method is 'constant' use this value for filling null.",
    is_list=False,
    is_optional=True,
    default_value="",
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


# current version 0.1
MODEL_MAX_MAJOR_VERSION = 0
MODEL_MAX_MINOR_VERSION = 1


def generate_rule_dict_constant(
    missing_value,
    fill_value_int,
    fill_value_float,
    fill_value_str,
    df: pd.DataFrame,
):
    dtypes_map = {
        "int": fill_value_int,
        "float": fill_value_float,
        "str": fill_value_str,
    }

    fill_value_dict = {}
    for dtype in dtypes_map.keys():
        for col in df.select_dtypes(
            include=[SUPPORTED_VTABLE_DATA_TYPE[dtype]]
        ).columns:
            fill_value_dict[col] = dtypes_map[dtype]
    rules_dict = {}
    rules_dict["fill_value_rule"] = fill_value_dict
    rules_dict["missing_value"] = missing_value
    return rules_dict


def generate_rule_dict(missing_value, data, strategy):
    # treat all missing_value as missing even if they were not
    data = data.replace({missing_value: np.nan})
    # this step will unify all np.nan, pd.NA and None values
    # just like these NA or null values entered arrow and all becoming null
    data = data.fillna(np.nan)

    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer.fit(data)
    fill_value_dict = {
        k: v for k, v in zip(imputer.feature_names_in_, imputer.statistics_)
    }
    fillna_rules = {"missing_value": missing_value, "fill_value_rule": fill_value_dict}
    return fillna_rules


def apply_fillna_rule_on_table(table: sc.Table, rules: Dict) -> sc.Table:
    fill_value_rules = rules["fill_value_rule"]
    missing_value = rules["missing_value"]
    is_nan = pd.isna(missing_value)
    for col_name in fill_value_rules:
        fill_value = fill_value_rules[col_name]
        col = table.column(col_name)
        cond = sc.is_nan(col) if is_nan else sc.equal(col, missing_value)
        # nan or specified values must be filled
        new_col = sc.if_else(cond, fill_value, col)
        # null values may still exist and must also be filled
        new_col = sc.coalesce(new_col, fill_value)
        table = table.set_column(table.column_names.index(col_name), col_name, new_col)
    return table


@fillna.eval_fn
def fillna_eval_fn(
    *,
    ctx,
    strategy,
    missing_value,
    missing_value_type,
    fill_value_int,
    fill_value_float,
    fill_value_str,
    input_dataset,
    input_dataset_fill_na_features,
    out_ds,
    out_rules,
):
    assert (
        input_dataset.type == DistDataType.VERTICAL_TABLE
    ), "only support vtable for now"

    assert strategy in SUPPORTED_FILL_NA_METHOD, f"unsupported strategy {strategy}"
    if missing_value_type == "general_na":
        missing_value = (
            np.nan
        )  # all pd.NA, np.nan and None are equivalent for arrow system
    else:
        try:
            missing_value = NA_SUPPORTED_TYPE_DICT[missing_value_type](missing_value)
        except ValueError:
            logging.warning(
                f"{missing_value} cannot be casted to {missing_value_type}, resort to general na only."
            )
            missing_value = np.nan

    def fit(data):
        if strategy in ["mean", "median"]:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            assert len(numeric_columns) == len(
                data.columns
            ), f"strategy {strategy} works only on numerical columns, select only numerical columns"
            fillna_rules = generate_rule_dict(missing_value, data, strategy)
        elif strategy == "most_frequent":
            fillna_rules = generate_rule_dict(missing_value, data, strategy)
        elif strategy == "constant":
            fillna_rules = generate_rule_dict_constant(
                missing_value,
                fill_value_int,
                fill_value_float,
                fill_value_str,
                data,
            )
        else:
            assert ValueError(f"Unsupported strategy {strategy}")
        return fillna_rules

    def fillna_fit_transform(trans_data):
        fillna_rules = fit(trans_data)
        trans_data = apply_fillna_rule_on_table(
            sc.Table.from_pandas(trans_data), fillna_rules
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
            load_ids=False,
            assert_one_party=False,
        )
    return {"out_ds": output_dd, "out_rules": model_dd}
