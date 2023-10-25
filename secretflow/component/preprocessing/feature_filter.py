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

import os

from secretflow.component.component import Component, IoType, TableColParam
from secretflow.component.data_utils import DistDataType, load_table
from secretflow.device.driver import wait
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable

feature_filter_comp = Component(
    "feature_filter",
    domain="preprocessing",
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

    in_meta = VerticalTable()
    in_ds.meta.Unpack(in_meta)

    out_meta = VerticalTable()
    out_meta.line_count = in_meta.line_count

    for s in in_meta.schemas:
        s_meta = TableSchema()
        for t, h in zip(s.feature_types, s.features):
            if h not in in_ds_drop_features:
                s_meta.feature_types.append(t)
                s_meta.features.append(h)
        s_meta.ids.extend(s.ids)
        s_meta.id_types.extend(s.id_types)
        s_meta.labels.extend(s.labels)
        s_meta.label_types.extend(s.label_types)
        out_meta.schemas.append(s_meta)

    out_dist = DistData()
    out_dist.CopyFrom(in_ds)
    out_dist.name = out_ds
    out_dist.meta.Pack(out_meta)

    # TODO: streaming
    with ctx.tracer.trace_running():
        ds = load_table(
            ctx, out_dist, load_features=True, load_ids=True, load_labels=True
        )
        out_path = {p: os.path.join(ctx.local_fs_wd, out_ds) for p in ds.partitions}
        wait(ds.to_csv(out_path, index=False))

    for i in range(len(out_dist.data_refs)):
        out_dist.data_refs[i].uri = out_ds

    return {"out_ds": out_dist}
