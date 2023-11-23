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

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import secretflow.compute as sc
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, SUPPORTED_VTABLE_DATA_TYPE

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
    default_value="mean",
    allowed_values=SUPPORTED_FILL_NA_METHOD,
)


fillna.bool_attr(
    name="fill_value_bool",
    desc="For bool type data. If method is 'constant' use this value for filling null.",
    is_list=False,
    is_optional=True,
    default_value=False,
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
    fill_value_bool, fill_value_int, fill_value_float, fill_value_str, df: pd.DataFrame
):
    rules_dict = {}
    dtypes_map = {
        "bool": fill_value_bool,
        "int": fill_value_int,
        "float": fill_value_float,
        "str": fill_value_str,
    }

    for dtype in dtypes_map.keys():
        for col in df.select_dtypes(
            include=[SUPPORTED_VTABLE_DATA_TYPE[dtype]]
        ).columns:
            rules_dict[col] = dtypes_map[dtype]
    return rules_dict


def generate_rule_dict(data, strategy):
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(data)

    fillna_rules = {
        k: v for k, v in zip(imputer.feature_names_in_, imputer.statistics_)
    }
    return fillna_rules


def apply_fillna_rule_on_table(table: sc.Table, rules: Dict) -> sc.Table:
    for col_name in rules:
        fill_value = rules[col_name]
        col = table.column(col_name)
        new_col = sc.coalesce(col, fill_value)
        table = table.set_column(table.column_names.index(col_name), col_name, new_col)
    return table


@fillna.eval_fn
def fillna_eval_fn(
    *,
    ctx,
    strategy,
    fill_value_bool,
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

    def fit(data):
        if strategy in ["mean", "median"]:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            assert len(numeric_columns) == len(
                data.columns
            ), f"strategy {strategy} works only on numerical columns, select only numerical columns"
            fillna_rules = generate_rule_dict(data, strategy)
        else:
            fillna_rules = generate_rule_dict_constant(
                fill_value_bool, fill_value_int, fill_value_float, fill_value_str, data
            )
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
