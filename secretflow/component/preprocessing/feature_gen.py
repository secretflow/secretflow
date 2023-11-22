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

import pandas as pd

import secretflow.compute as sc
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, load_table
from secretflow.component.preprocessing.core.table_utils import (
    v_preprocessing_transform,
)

feature_gen_comp = Component(
    "feature_gen",
    domain="preprocessing",
    version="0.0.2",
    desc="Generate a new feature based on one original feature and another feature or constant.",
)
feature_gen_comp.str_attr(
    name="binary_op",
    desc="What kind of binary operation we want to do, currently only supports +, -, *, /",
    is_list=False,
    is_optional=True,
    allowed_values=["+", "-", "*", "/"],
    default_value="+",
)

feature_gen_comp.float_attr(
    name="constant",
    desc="If we have only one feature, we assume, new_feature = binary_op(f1, constant).",
    is_list=False,
    is_optional=True,
    default_value=0,
)

feature_gen_comp.str_attr(
    name="new_feature_name",
    desc="Name of the newly generated feature.",
    is_list=False,
    is_optional=False,
)

feature_gen_comp.bool_attr(
    name="as_label",
    desc="If True, the generated feature will be marked as label in schema.",
    is_list=False,
    is_optional=True,
    default_value=False,
)


feature_gen_comp.io(
    io_type=IoType.INPUT,
    name="in_ds",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="features",
            desc="Feature(s) to operate on. If two features exist, we assume new_feature = binary_op(f1, f2). Currently two features must be from the same party.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=2,
        )
    ],
)


feature_gen_comp.io(
    io_type=IoType.OUTPUT,
    name="out_ds",
    desc="Output vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)

feature_gen_comp.io(
    io_type=IoType.OUTPUT,
    name="out_rules",
    desc="feature gen rule",
    types=[DistDataType.PREPROCESSING_RULE],
    col_params=None,
)


OP_MAP = {
    "+": sc.add,
    "-": sc.subtract,
    "*": sc.multiply,
    "/": sc.divide,
}


@feature_gen_comp.eval_fn
def feature_gen_eval_fn(
    *,
    ctx,
    in_ds,
    in_ds_features,
    binary_op,
    constant,
    as_label,
    new_feature_name,
    out_ds,
    out_rules,
):
    assert in_ds.type == DistDataType.VERTICAL_TABLE, "only support vtable for now"
    head = load_table(
        ctx, in_ds, load_features=True, load_labels=True, load_ids=True, nrows=1
    )
    if new_feature_name in head.columns:
        load_columns = in_ds_features + [new_feature_name]
    else:
        load_columns = in_ds_features

    def _compute_new_table(
        df: pd.DataFrame,
    ) -> sc.Table:
        df_sc = sc.Table.from_pandas(df)
        assert (
            in_ds_features[0] in df.columns
        ), "we should not load tables that need no action."
        arg_0 = df_sc.column(in_ds_features[0])
        arg_1 = None
        if len(in_ds_features) > 1:
            arg_1 = df_sc.column(in_ds_features[1])
        else:
            arg_1 = constant

        if new_feature_name in df.columns:
            df_sc = df_sc.set_column(
                df_sc.column_names.index(new_feature_name),
                new_feature_name,
                OP_MAP[binary_op](arg_0, arg_1),
            )
        else:
            df_sc = df_sc.append_column(
                name=new_feature_name, array=OP_MAP[binary_op](arg_0, arg_1)
            )

        return df_sc, [new_feature_name] if as_label else [], None

    (output_dd, model_dd, _) = v_preprocessing_transform(
        ctx,
        in_ds,
        load_columns,
        _compute_new_table,
        out_ds,
        out_rules,
        "Feature Generate",
    )

    return {
        "out_rules": model_dd,
        "out_ds": output_dd,
    }
