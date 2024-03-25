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

import json

from secretflow.component.component import CompEvalError, Component, IoType
from secretflow.component.data_utils import DistDataType, model_dumps, model_loads
from secretflow.device.device.spu import SPU
from secretflow.spec.extend.data_pb2 import DeviceObjectCollection
from secretflow.spec.v1.data_pb2 import DistData

IDENTITY_SUPPORTED_TYPES = [
    DistDataType.SS_GLM_MODEL,
    DistDataType.SGB_MODEL,
    DistDataType.SS_XGB_MODEL,
    DistDataType.SS_SGD_MODEL,
    DistDataType.BIN_RUNNING_RULE,
    DistDataType.PREPROCESSING_RULE,
    DistDataType.READ_DATA,
]
identity = Component(
    "identity",
    domain="io",
    version="0.0.1",
    desc="map any input to output",
)

identity.io(
    io_type=IoType.INPUT,
    name="input_data",
    desc="Input data",
    types=IDENTITY_SUPPORTED_TYPES,
)

identity.io(
    io_type=IoType.OUTPUT,
    name="output_data",
    desc="Output data",
    types=IDENTITY_SUPPORTED_TYPES,
)


@identity.eval_fn
def identity_eval_fn(*, ctx, input_data: DistData, output_data):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    model_meta = DeviceObjectCollection()
    assert input_data.meta.Unpack(model_meta)

    model_info = json.loads(model_meta.public_info)

    objs, public_info = model_loads(
        ctx,
        input_data,
        model_info["major_version"],
        model_info["minor_version"],
        input_data.type,
        spu=spu,
    )
    output_data_dd = model_dumps(
        ctx,
        input_data.name,
        input_data.type,
        model_info["major_version"],
        model_info["minor_version"],
        objs,
        public_info,
        output_data,
        input_data.system_info,
    )
    return {"output_data": output_data_dd}
