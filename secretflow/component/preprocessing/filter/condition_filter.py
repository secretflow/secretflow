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

import math
from typing import Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, extract_data_infos
from secretflow.component.dataframe import (
    CompDataFrame,
    StreamingReader,
    StreamingWriter,
)
from secretflow.device.device.pyu import PYU

condition_filter_comp = Component(
    "condition_filter",
    domain="data_filter",
    version="0.0.2",
    desc="""Filter the table based on a single column's values and condition.
    Warning: the party responsible for condition filtering will directly send the sample distribution to other participants.
    Malicious participants can obtain the distribution of characteristics by repeatedly calling with different filtering values.
    Audit the usage of this component carefully.
    """,
)

condition_filter_comp.str_attr(
    name="comparator",
    desc="Comparator to use for comparison. Must be one of '==','<','<=','>','>=','IN','NOTNULL' ",
    is_list=False,
    is_optional=False,
    allowed_values=['==', '<', '<=', '>', '>=', 'IN', 'NOTNULL'],
)

condition_filter_comp.str_attr(
    name="bound_value",
    desc="Input a value for comparison; if the comparison condition is IN, you can input multiple values separated by ','; if the comparison condition is NOTNULL, the input is not needed.",
    is_optional=True,
    is_list=False,
    default_value="",
)

condition_filter_comp.float_attr(
    name="float_epsilon",
    desc="Epsilon value for floating point comparison. WARNING: due to floating point representation in computers, set this number slightly larger if you want filter out the values exactly at desired boundary. for example, abs(1.001 - 1.002) is slightly larger than 0.001, and therefore may not be filter out using == and epsilson = 0.001",
    is_list=False,
    is_optional=True,
    lower_bound=0,
    lower_bound_inclusive=True,
    default_value=0.000001,
)

condition_filter_comp.io(
    io_type=IoType.INPUT,
    name="in_ds",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="features",
            desc="Feature(s) to operate on.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        )
    ],
)

condition_filter_comp.io(
    io_type=IoType.OUTPUT,
    name="out_ds",
    desc="Output vertical table that satisfies the condition.",
    types=[DistDataType.VERTICAL_TABLE],
)

condition_filter_comp.io(
    io_type=IoType.OUTPUT,
    name="out_ds_else",
    desc="Output vertical table that does not satisfies the condition.",
    types=[DistDataType.VERTICAL_TABLE],
)

comparator_mapping = {
    '==': lambda x, y: pc.equal(x, y),
    '<': lambda x, y: pc.less(x, y),
    '<=': lambda x, y: pc.less_equal(x, y),
    '>': lambda x, y: pc.greater(x, y),
    '>=': lambda x, y: pc.greater_equal(x, y),
    'IN': lambda x, y: pc.is_in(x, y),
    'NOTNULL': lambda x, _: pc.is_valid(x),
}


def is_close(a, b, epsilon=1e-7):
    return pc.less_equal(pc.abs(pc.subtract(a, b)), epsilon)


def approx_is_in(array, values, epsilon=1e-7):
    conditions = [is_close(array, value, epsilon) for value in values]
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition = pc.or_(combined_condition, condition)
    return combined_condition


def get_compare_func(comparator: str, is_float: bool, epsilon: float):
    if is_float:
        if comparator == "==":
            return lambda x, y: is_close(x, y, epsilon)
        elif comparator == "IN":
            return lambda x, y: approx_is_in(x, y, epsilon)

    return comparator_mapping[comparator]


def apply(
    df: CompDataFrame, owner: PYU, name: str, value, compare_fn
) -> tuple[CompDataFrame, CompDataFrame]:
    def _fit(table: pa.Table) -> pa.Table:
        filter = compare_fn(table[name], value)
        return pc.fill_null(filter, False)

    def _transform(
        df: pa.Table, selection: pa.ChunkedArray
    ) -> Tuple[pa.Table, pa.Table]:
        return df.filter(selection), df.filter(pc.invert(selection))

    selection = owner(_fit)(df.partitions[owner].data.data)

    selected_df = df.copy()
    else_df = df.copy()
    for pyu in df.partitions:
        selected_data, else_data = pyu(_transform)(df.data(pyu), selection.to(pyu))
        selected_df.set_data(selected_data)
        else_df.set_data(else_data)

    return selected_df, else_df


@condition_filter_comp.eval_fn
def condition_filter_comp_eval_fn(
    *,
    ctx,
    comparator,
    bound_value,
    float_epsilon,
    in_ds,
    in_ds_features,
    out_ds,
    out_ds_else,
):
    data_infos = extract_data_infos(
        in_ds,
        load_features=True,
        load_ids=True,
        load_labels=True,
        col_selects=in_ds_features,
    )
    assert len(data_infos) == 1, f"cannot find feature, {in_ds_features}"
    owner = next(iter(data_infos.keys()))
    np_type = data_infos[owner].dtypes[in_ds_features[0]]
    if pd.api.types.is_float_dtype(np_type):
        value_type = float
    elif pd.api.types.is_string_dtype(np_type):
        value_type = str
    elif pd.api.types.is_integer_dtype(np_type):
        value_type = int
    else:
        raise ValueError(f"only support FLOAT, STRING for now, but got type<{np_type}>")

    value = None
    if comparator != "NOTNULL":
        bound_value_list = bound_value.split(",")
        if bound_value == "":
            raise ValueError(f"bound_value is empty")
        values = [value_type(val) for val in bound_value_list]
        value = values if comparator == "IN" else values[0]

    is_float = value_type is float
    fn = get_compare_func(comparator, is_float, float_epsilon)

    # Load data from train_dataset
    reader = StreamingReader.from_distdata(
        ctx, in_ds, load_features=True, load_ids=True, load_labels=True
    )
    out_writer = StreamingWriter(ctx, out_ds)
    else_writer = StreamingWriter(ctx, out_ds_else)
    bound_value_list = bound_value.split(",")

    def write(writer, ds):
        if math.prod(ds.shape):
            writer.write(ds)

    with out_writer, else_writer, ctx.tracer.trace_running():
        for batch in reader:
            ds, else_ds = apply(batch, PYU(owner), in_ds_features[0], value, fn)
            write(out_writer, ds)
            write(else_writer, else_ds)

    assert (
        out_writer.line_count
    ), f"empty dataset is not allowed, yet the table satisfied the condition is empty, \
    skip this condition filter step and use alternative pipeline please."

    assert (
        else_writer.line_count
    ), f"empty dataset is not allowed, yet the table not satisfied the condition is empty, \
    skip this condition filter step and use alternative pipeline please."

    return {
        "out_ds": out_writer.to_distdata(),
        "out_ds_else": else_writer.to_distdata(),
    }
