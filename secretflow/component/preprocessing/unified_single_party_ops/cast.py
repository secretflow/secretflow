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

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

import secretflow.compute as sc
from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, extract_table_header
from secretflow.component.preprocessing.core.table_utils import (
    v_preprocessing_transform,
)
from secretflow.spec.v1.data_pb2 import DistData

CAST_VERSION = "0.0.1"

cast_comp = Component(
    "cast",
    domain="preprocessing",
    version=CAST_VERSION,
    desc="For conversion between basic data types, such as converting float to string.",
)

_SUPPORTED_CAST_TYPE = ["integer", "float", "string"]

cast_comp.union_attr_group(
    name="astype",
    desc="single-choice, options available are string, integer, float",
    group=[
        cast_comp.union_selection_attr(
            name="integer",
            desc="integer",
        ),
        cast_comp.union_selection_attr(
            name="float",
            desc="float",
        ),
        cast_comp.union_selection_attr(
            name="string",
            desc="string",
        ),
    ],
)

cast_comp.io(
    io_type=IoType.INPUT,
    name="input_ds",
    desc="The input table",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="columns",
            desc="Multiple-choice, options available are string, integer, float, boolean",
        )
    ],
)

cast_comp.io(
    io_type=IoType.OUTPUT,
    name="output_ds",
    desc="The output table",
    types=[DistDataType.VERTICAL_TABLE],
)

cast_comp.io(
    io_type=IoType.OUTPUT,
    name="output_rules",
    desc="The output rules",
    types=[DistDataType.PREPROCESSING_RULE],
)


def _apply_rules_on_table(table: sc.Table, target: str) -> sc.Table:
    is_numeric_target = target == 'int' or target == 'float'
    if target == 'int':
        pa_type = pa.int64()
    elif target == 'float':
        pa_type = pa.float64()
    elif target == 'str':
        pa_type = pa.string()
    else:
        raise ValueError(f"unsupported target type {target}")

    columns = table.column_names
    for col_name in columns:
        col = table.column(col_name)
        if col.dtype == pa_type:
            continue
        try:
            if pa.types.is_string(col.dtype) and is_numeric_target:
                col = sc.utf8_trim(col, characters=" \t\n\v\f\r\"'")
            options = pc.CastOptions.unsafe(pa_type)
            new_col = sc.cast(col, options=options)
            table = table.set_column(
                table.column_names.index(col_name), col_name, new_col
            )
        except Exception as e:
            raise ValueError(f"cast {col_name} failed, {e}")
    return table


@cast_comp.eval_fn
def cast_eval_fn(
    *, ctx, astype, input_ds: DistData, input_ds_columns, output_ds, output_rules
):
    astype = astype.lower()
    assert astype in _SUPPORTED_CAST_TYPE, f"not support type {astype}"

    astype_map = {
        "integer": "int",
        "float": "float",
        "string": "str",
    }

    astype = astype_map[astype]

    label_info, _ = extract_table_header(
        input_ds, load_features=False, load_labels=True
    )
    label_set = set()
    for kv in label_info.values():
        label_set.update(kv.keys())

    def _fit_transform(trans_data: pd.DataFrame):
        table = _apply_rules_on_table(sc.Table.from_pandas(trans_data), astype)
        columns = list(trans_data.columns)
        labels = set(columns).intersection(label_set)
        return table, list(labels), {}

    (output_dd, model_dd, _) = v_preprocessing_transform(
        ctx,
        input_ds,
        input_ds_columns,
        _fit_transform,
        output_ds,
        output_rules,
        "cast",
        load_ids=False,
        assert_one_party=False,
    )

    return {"output_ds": output_dd, "output_rules": model_dd}
