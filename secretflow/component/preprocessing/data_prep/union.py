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

from secretflow.component.component import Component, IoType
from secretflow.component.data_utils import DistDataType, extract_data_infos
from secretflow.component.dataframe import (
    CompDataFrame,
    StreamingReader,
    StreamingWriter,
)
from secretflow.data.core import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.pyu import PYU
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable

union_comp = Component(
    "union",
    domain="data_prep",
    version="0.0.1",
    desc="Perform a horizontal merge of two data tables, supporting the individual table or vertical table on the same node.",
)

union_comp.io(
    io_type=IoType.INPUT,
    name="input1",
    desc="The first input table",
    types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    col_params=None,
)

union_comp.io(
    io_type=IoType.INPUT,
    name="input2",
    desc="The second input table",
    types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
    col_params=None,
)

union_comp.io(
    io_type=IoType.OUTPUT,
    name="output_ds",
    desc="Output table",
    types=[DistDataType.INDIVIDUAL_TABLE, DistDataType.VERTICAL_TABLE],
)


@union_comp.eval_fn
def union_eval_fn(
    *,
    ctx,
    input1: DistData,
    input2: DistData,
    output_ds,
):
    assert (
        input1.type.lower() == input2.type.lower()
    ), f"input type not match, {input1.type}, {input2.type}"

    table1_infos = extract_data_infos(
        input1,
        load_features=True,
        load_labels=True,
        load_ids=True,
    )
    table2_infos = extract_data_infos(
        input2,
        load_features=True,
        load_labels=True,
        load_ids=True,
    )

    assert set(table1_infos.keys()) == set(table2_infos.keys()), f"partitions not match"
    for p in table1_infos:
        assert table1_infos[p].mate_equal(table2_infos[p]), f"table meta info missmatch"

    reader1 = StreamingReader.from_distdata(
        ctx,
        input1,
        load_features=True,
        load_labels=True,
        load_ids=True,
    )
    reader2 = StreamingReader.from_distdata(
        ctx,
        input2,
        load_features=True,
        load_labels=True,
        load_ids=True,
    )

    writer = StreamingWriter(ctx, output_ds)

    with writer, ctx.tracer.trace_io():
        for batch in reader1:
            writer.write(batch)
        for batch in reader2:
            writer.write(batch)

    return {"output_ds": writer.to_distdata()}
