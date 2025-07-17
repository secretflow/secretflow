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

import pyarrow as pa
import pyarrow.compute as pc

from secretflow.component.core import (
    Component,
    CompVDataFrame,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Output,
    VTable,
    VTableFieldKind,
    register,
)
from secretflow.device import PYU
from secretflow.utils.errors import InvalidArgumentError, InvalidStateError

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
    df: CompVDataFrame, owner: PYU, name: str, value, compare_fn
) -> tuple[CompVDataFrame, CompVDataFrame]:
    def _fit(table: pa.Table) -> pa.Table:
        filter = compare_fn(table[name], value)
        return pc.fill_null(filter, False)

    def _transform(
        df: pa.Table, selection: pa.ChunkedArray
    ) -> Tuple[pa.Table, pa.Table]:
        return df.filter(selection), df.filter(pc.invert(selection))

    selection = owner(_fit)(df.partitions[owner].data)

    selected_df = CompVDataFrame({}, df.system_info)
    else_df = CompVDataFrame({}, df.system_info)
    for pyu in df.partitions:
        selected_data, else_data = pyu(_transform)(df.data(pyu), selection.to(pyu))
        selected_df.set_data(selected_data)
        else_df.set_data(else_data)

    return selected_df, else_df


@register(domain="data_filter", version="1.0.0")
class ConditionFilter(Component):
    '''
    Filter the table based on a single column's values and condition.
    Warning: the party responsible for condition filtering will directly send the sample distribution to other participants.
    Malicious participants can obtain the distribution of characteristics by repeatedly calling with different filtering values.
    Audit the usage of this component carefully.
    '''

    comparator: str = Field.attr(
        desc="Comparator to use for comparison. Must be one of '==','<','<=','>','>=','IN','NOTNULL' ",
        choices=['==', '<', '<=', '>', '>=', 'IN', 'NOTNULL'],
    )
    bound_value: str = Field.attr(
        desc="Input a value for comparison; if the comparison condition is IN, you can input multiple values separated by ','; if the comparison condition is NOTNULL, the input is not needed.",
        default="",
    )
    float_epsilon: float = Field.attr(
        desc="Epsilon value for floating point comparison. WARNING: due to floating point representation in computers, set this number slightly larger if you want filter out the values exactly at desired boundary. for example, abs(1.001 - 1.002) is slightly larger than 0.001, and therefore may not be filter out using == and epsilson = 0.001",
        default=0.000001,
        bound_limit=Interval.closed(0, None),
    )
    feature: str = Field.table_column_attr(
        "input_ds",
        desc="Feature to operate on.",
    )
    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output vertical table that satisfies the condition.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds_else: Output = Field.output(
        desc="Output vertical table that does not satisfies the condition.",
        types=[DistDataType.VERTICAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        info = VTable.from_distdata(self.input_ds, columns=[self.feature])
        info.check_kinds(VTableFieldKind.FEATURE)
        assert len(info.parties) == 1, f"cannot find feature, {self.feature}"

        owner = next(iter(info.parties.keys()))
        field = info.get_schema(0).get_field(0)
        if field.type.is_float():
            value_type = float
        elif field.type.is_string():
            value_type = str
        elif field.type.is_integer():
            value_type = int
        else:
            raise InvalidArgumentError(
                "only support FLOAT, STRING for now", detail={"type": field.type}
            )

        value = None
        if self.comparator != "NOTNULL":
            bound_value_list = self.bound_value.split(",")
            if self.bound_value == "":
                raise InvalidArgumentError("bound_value is empty")
            values = [value_type(val) for val in bound_value_list]
            value = values if self.comparator == "IN" else values[0]

        is_float = value_type is float
        fn = get_compare_func(self.comparator, is_float, self.float_epsilon)
        bound_value_list = self.bound_value.split(",")

        # Load data from train_dataset
        reader = CompVDataFrameReader(ctx.storage, ctx.tracer, self.input_ds)
        out_writer = CompVDataFrameWriter(ctx.storage, ctx.tracer, self.output_ds.uri)
        else_writer = CompVDataFrameWriter(
            ctx.storage, ctx.tracer, self.output_ds_else.uri
        )

        def write(writer, ds):
            if math.prod(ds.shape):
                writer.write(ds)

        with out_writer, else_writer:
            for batch in reader:
                ds, else_ds = apply(batch, PYU(owner), self.feature, value, fn)
                write(out_writer, ds)
                write(else_writer, else_ds)

        if out_writer.line_count == 0 or else_writer.line_count == 0:
            raise InvalidStateError(
                message="empty dataset is not allowed, skip this condition filter step and use alternative pipeline please.",
                detail={
                    "out_line_count": out_writer.line_count,
                    "else_line_count": else_writer.line_count,
                },
            )

        out_writer.dump_to(self.output_ds)
        else_writer.dump_to(self.output_ds_else)
