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
    Interval,
    Output,
    VTable,
    register,
)


@register(domain="data_filter", version="1.0.0")
class FeatureFilter(Component):
    '''
    Drop features from the dataset.
    '''

    drop_features: list[str] = Field.table_column_attr(
        "input_ds",
        desc="Features to drop.",
        limit=Interval.closed(1, None),
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )
    output_ds: Output = Field.output(
        desc="Output vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
    )

    def evaluate(self, ctx: Context):
        in_tbl = VTable.from_distdata(self.input_ds)
        in_tbl = in_tbl.drop(self.drop_features)
        reader = CompVDataFrameReader(ctx.storage, ctx.tracer, in_tbl)
        writer = CompVDataFrameWriter(ctx.storage, ctx.tracer, self.output_ds.uri)
        with writer:
            for df in reader:
                writer.write(df)

        writer.dump_to(self.output_ds)
