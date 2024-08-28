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

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType
from secretflow.component.dataframe import (
    CompDataFrame,
    StreamingReader,
    StreamingWriter,
)
from secretflow.device.driver import wait
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable

feature_filter_comp = Component(
    "feature_filter",
    domain="data_filter",
    version="0.0.1",
    desc="Drop features from the dataset.",
)

feature_filter_comp.io(
    io_type=IoType.INPUT,
    name="in_ds",
    desc="Input vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[TableColParam(name="drop_features", desc="Features to drop.")],
)

feature_filter_comp.io(
    io_type=IoType.OUTPUT,
    name="out_ds",
    desc="Output vertical table.",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=None,
)


@feature_filter_comp.eval_fn
def feature_filter_eval_fn(*, ctx, in_ds, in_ds_drop_features, out_ds):
    assert in_ds.type == DistDataType.VERTICAL_TABLE, "only support vtable for now"

    reader = StreamingReader.from_distdata(
        ctx, in_ds, load_features=True, load_ids=True, load_labels=True
    )
    writer = StreamingWriter(ctx, out_ds)
    with writer, ctx.tracer.trace_running():
        for batch in reader:
            writer.write(batch.drop(in_ds_drop_features))

    return {"out_ds": writer.to_distdata()}
