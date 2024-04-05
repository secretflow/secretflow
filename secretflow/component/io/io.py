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

from google.protobuf.json_format import Parse

from secretflow.component.component import CompEvalError, Component, IoType
from secretflow.component.data_utils import (
    DistDataType,
    any_pyu_from_spu_config,
    generate_random_string,
    model_dumps,
    model_loads,
)
from secretflow.component.io.core.bins.bins import (
    bin_rule_from_pb_and_old_rule,
    bin_rule_to_pb,
)
from secretflow.component.io.core.linear_model.ss_glm import (
    ss_glm_from_pb_and_old_model,
    ss_glm_to_linear_model_pb,
)
from secretflow.component.ml.linear.ss_glm import (
    MODEL_MAX_MAJOR_VERSION as GLM_MAX_MAJOR_VERSION,
)
from secretflow.component.ml.linear.ss_glm import (
    MODEL_MAX_MINOR_VERSION as GLM_MAX_MINOR_VERSION,
)
from secretflow.component.preprocessing.binning.vert_binning import (
    BINNING_RULE_MAX_MAJOR_VERSION,
    BINNING_RULE_MAX_MINOR_VERSION,
)
from secretflow.device.device.spu import SPU
from secretflow.spec.extend.bin_data_pb2 import Bins
from secretflow.spec.extend.linear_model_pb2 import LinearModel
from secretflow.spec.v1.data_pb2 import DistData

io_read_data = Component(
    "read_data",
    domain="io",
    version="0.0.1",
    desc="read model or rules from sf cluster",
)

io_read_data.io(
    io_type=IoType.INPUT,
    name="input_dd",
    desc="Input dist data",
    types=[
        DistDataType.BIN_RUNNING_RULE,
        DistDataType.SS_GLM_MODEL,
        # add others module or rules support here
    ],
)

io_read_data.io(
    io_type=IoType.OUTPUT,
    name="output_data",
    desc="Output rules or models in DistData.meta",
    types=[DistDataType.READ_DATA],
    col_params=None,
)


@io_read_data.eval_fn
def io_read_data_eval_fn(*, ctx, input_dd, output_data):
    if input_dd.type == DistDataType.BIN_RUNNING_RULE:
        model_objs, public_info = model_loads(
            ctx,
            input_dd,
            BINNING_RULE_MAX_MAJOR_VERSION,
            BINNING_RULE_MAX_MINOR_VERSION,
            DistDataType.BIN_RUNNING_RULE,
        )

        read_data = bin_rule_to_pb(model_objs, public_info)
    elif input_dd.type == DistDataType.SS_GLM_MODEL:
        if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
            raise CompEvalError("spu config is not found.")
        if len(ctx.spu_configs) > 1:
            raise CompEvalError("only support one spu")
        spu_config = next(iter(ctx.spu_configs.values()))

        cluster_def = spu_config["cluster_def"].copy()

        # forced to use 128 ring size & 40 fxp
        cluster_def["runtime_config"]["field"] = "FM128"
        cluster_def["runtime_config"]["fxp_fraction_bits"] = 40

        spu = SPU(cluster_def, spu_config["link_desc"])
        model_objs, public_info = model_loads(
            ctx,
            input_dd,
            GLM_MAX_MAJOR_VERSION,
            GLM_MAX_MINOR_VERSION,
            DistDataType.SS_GLM_MODEL,
            spu=spu,
        )
        read_data = ss_glm_to_linear_model_pb(model_objs, public_info)
    #     # add others module or rules support here
    #     pass
    else:
        raise AttributeError(f"unknown model/rules type {input_dd.type}")

    output_data_dd = DistData(
        name=output_data,
        type=str(DistDataType.READ_DATA),
        system_info=input_dd.system_info,
    )
    output_data_dd.meta.Pack(read_data)

    return {"output_data": output_data_dd}


io_write_data = Component(
    "write_data",
    domain="io",
    version="0.0.1",
    desc="write model or rules back to sf cluster",
)

io_write_data.str_attr(
    name="write_data",
    desc="rule or model protobuf by json formate",
    is_list=False,
    is_optional=False,
)

CURRENT_SUPPORTED_TYPES = [
    str(DistDataType.BIN_RUNNING_RULE),
    str(DistDataType.SS_GLM_MODEL),
]
io_write_data.str_attr(
    name="write_data_type",
    desc="which rule or model is writing",
    is_list=False,
    is_optional=True,
    default_value=CURRENT_SUPPORTED_TYPES[0],
    allowed_values=CURRENT_SUPPORTED_TYPES,
)

io_write_data.io(
    io_type=IoType.INPUT,
    name="input_dd",
    desc="Input dist data. Rule reconstructions may need hidden info in original rule for security considerations.",
    types=[
        DistDataType.BIN_RUNNING_RULE,
        DistDataType.SS_GLM_MODEL,
        # add others module or rules support here
    ],
)

io_write_data.io(
    io_type=IoType.OUTPUT,
    name="output_model",
    desc="Output rules or models in sf cluster format",
    types=[
        DistDataType.BIN_RUNNING_RULE,
        DistDataType.SS_GLM_MODEL,
        # add others module or rules support here
    ],
    col_params=None,
)


@io_write_data.eval_fn
def io_write_data_eval_fn(*, ctx, write_data, write_data_type, input_dd, output_model):
    if write_data_type == str(DistDataType.BIN_RUNNING_RULE):
        model_objs, public_info = model_loads(
            ctx,
            input_dd,
            BINNING_RULE_MAX_MAJOR_VERSION,
            BINNING_RULE_MAX_MINOR_VERSION,
            DistDataType.BIN_RUNNING_RULE,
        )
        bin_rules = Parse(write_data, Bins())
        rule_objs = bin_rule_from_pb_and_old_rule(model_objs, public_info, bin_rules)

        info_dict = public_info
        info_dict["model_hash"] = generate_random_string(model_objs[0].device)
        output_model_dd = model_dumps(
            ctx,
            "bin_rule",
            DistDataType.BIN_RUNNING_RULE,
            BINNING_RULE_MAX_MAJOR_VERSION,
            BINNING_RULE_MAX_MINOR_VERSION,
            rule_objs,
            info_dict,
            output_model,
            input_dd.system_info,
        )
    elif write_data_type == str(DistDataType.SS_GLM_MODEL):
        if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
            raise CompEvalError("spu config is not found.")
        if len(ctx.spu_configs) > 1:
            raise CompEvalError("only support one spu")
        spu_config = next(iter(ctx.spu_configs.values()))

        cluster_def = spu_config["cluster_def"].copy()

        # forced to use 128 ring size & 40 fxp
        cluster_def["runtime_config"]["field"] = "FM128"
        cluster_def["runtime_config"]["fxp_fraction_bits"] = 40

        spu = SPU(cluster_def, spu_config["link_desc"])
        model_objs, public_info = model_loads(
            ctx,
            input_dd,
            GLM_MAX_MAJOR_VERSION,
            GLM_MAX_MINOR_VERSION,
            DistDataType.SS_GLM_MODEL,
            spu=spu,
        )
        new_model_objs = ss_glm_from_pb_and_old_model(
            model_objs, public_info, Parse(write_data, LinearModel())
        )
        info_dict = json.loads(public_info)
        info_dict["model_hash"] = generate_random_string(
            any_pyu_from_spu_config(cluster_def)
        )
        output_model_dd = model_dumps(
            ctx,
            "ss_glm",
            DistDataType.SS_GLM_MODEL,
            GLM_MAX_MAJOR_VERSION,
            GLM_MAX_MINOR_VERSION,
            new_model_objs,
            json.dumps(info_dict),
            output_model,
            input_dd.system_info,
        )
    #     # add others module or rules support here
    #     pass
    else:
        raise AttributeError(f"unknown model/rules type {write_data_type}")

    return {"output_model": output_model_dd}
