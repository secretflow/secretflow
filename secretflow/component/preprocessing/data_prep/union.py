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

from secretflow.component.core import (
    Component,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    Context,
    DistDataType,
    Field,
    Input,
    Output,
    VTable,
    register,
)
from secretflow.utils.errors import InvalidArgumentError


@register(domain="data_prep", version="1.0.0")
class Union(Component):
    '''
    Perform a horizontal merge of two data tables, supporting the individual table or vertical table on the same node.
    '''

    input_ds1: Input = Field.input(
        desc="The first input table",
        types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    )
    input_ds2: Input = Field.input(
        desc="The second input table",
        types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    )

    output_ds: Output = Field.output(
        desc="Output table",
        types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        if self.input_ds1.type.lower() != self.input_ds2.type.lower():
            raise InvalidArgumentError(
                "input type mismatch",
                detail={"type1": self.input_ds1.type, "type2": self.input_ds2.type},
            )

        tbl1 = VTable.from_distdata(self.input_ds1)
        tbl2 = VTable.from_distdata(self.input_ds2)
        if tbl1.schemas != tbl2.schemas:
            raise InvalidArgumentError("input schema mismatch")

        reader1 = CompVDataFrameReader(ctx.storage, ctx.tracer, tbl1)
        reader2 = CompVDataFrameReader(ctx.storage, ctx.tracer, tbl2)
        writer = CompVDataFrameWriter(ctx.storage, ctx.tracer, self.output_ds.uri)
        with reader1, reader2, writer:
            for df in reader1:
                writer.write(df)
            for df in reader2:
                writer.write(df)
        writer.dump_to(self.output_ds)
