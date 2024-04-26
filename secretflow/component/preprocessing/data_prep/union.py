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
from secretflow.component.data_utils import (
    DistDataType,
    VerticalTableWrapper,
    dump_table,
    load_table,
)
from secretflow.data.core import partition
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device.device.pyu import PYU
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable

union_comp = Component(
    "union",
    domain="data_prep",
    version="0.0.1",
    desc="Merge two datasets in vertical axis, the table type can be individual or vertical but must be the same.",
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
    output_ds: DistData,
):
    assert (
        input1.type.lower() == input2.type.lower()
    ), f"input type not match, {input1.type}, {input2.type}"

    tbl1 = load_table(
        ctx,
        input1,
        load_features=True,
        load_labels=True,
        load_ids=True,
    )
    tbl2 = load_table(
        ctx,
        input2,
        load_features=True,
        load_labels=True,
        load_ids=True,
    )

    assert len(tbl1.partitions) == len(tbl2.partitions), f"partitions not match"

    def _apply(df1: pd.DataFrame, df2: pd.DataFrame):
        return pd.concat([df1, df2], ignore_index=True)

    with ctx.tracer.trace_running():
        out_partitions = {}
        for device, party1 in tbl1.partitions.items():
            party2 = tbl2.partitions.get(device)
            assert party2 is not None, f"party not match {device}"
            assert (
                party1.columns == party2.columns
            ), f"columns not match, {party1.columns}, {party2.columns}"
            out_data = device(_apply)(party1.data, party2.data)
            out_partitions[device] = partition(out_data)

    out_aligned = tbl1.aligned and tbl2.aligned
    out_df = VDataFrame(partitions=out_partitions, aligned=out_aligned)

    if input1.type == DistDataType.VERTICAL_TABLE:
        meta = VerticalTableWrapper.from_dist_data(input1, out_df.shape[0])
    else:
        meta = IndividualTable()
        input1.meta.Unpack(meta)

    output_dd = dump_table(
        ctx,
        vdata=out_df,
        uri=output_ds,
        meta=meta,
        system_info=input1.system_info,
    )

    return {"output_ds": output_dd}
