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

import base64
import uuid
from typing import List

from google.protobuf import json_format

from secretflow.component.component import CompEvalError, Component, IoType
from secretflow.component.data_utils import DistDataType, extract_table_header
from secretflow.device import PYU, reveal
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam

from .serving_utils import (
    GraphBuilderManager,
    preprocessing_converter,
    train_model_converter,
)

model_export_comp = Component(
    "model_export",
    domain="model",
    version="0.0.1",
    desc="",
)

model_export_comp.str_attr(
    name="model_name",
    desc="model's name",
    is_list=False,
    is_optional=False,
)

model_export_comp.str_attr(
    name="model_desc",
    desc="Describe what the model does",
    is_list=False,
    is_optional=True,
    default_value="",
)

model_export_comp.str_attr(
    name="input_datasets",
    desc="all components' input data id",
    is_list=True,
    is_optional=False,
)

model_export_comp.str_attr(
    name="output_datasets",
    desc="all components' output data id",
    is_list=True,
    is_optional=False,
)

model_export_comp.str_attr(
    name="component_eval_params",
    desc="all components' params in json format",
    is_list=True,
    is_optional=False,
)

model_export_comp.io(
    io_type=IoType.OUTPUT,
    name="package_output",
    desc="output tar package uri",
    types=[DistDataType.SERVING_MODEL],
)


def get_comp_def(domain: str, name: str, version: str):
    from secretflow.component.entry import get_comp_def

    return get_comp_def(domain, name, version)


def get_init_pyus(input_datasets, component_eval_params) -> List[PYU]:
    first_comp = component_eval_params[0]
    comp_def = get_comp_def(first_comp.domain, first_comp.name, first_comp.version)
    dist_datas = input_datasets[: len(comp_def.inputs)]
    v_tables = [d for d in dist_datas if d.type == DistDataType.VERTICAL_TABLE]
    assert len(v_tables) == 1, "only support one v table input for now"
    v_headers = extract_table_header(
        v_tables[0], load_features=True, load_ids=True, load_labels=True
    )
    assert len(v_headers) > 0

    return [PYU(p) for p in v_headers]


class CompConverter:
    def __init__(self, ctx, pyus: List[PYU], spu_config):
        self.ctx = ctx
        self.graph_builders = GraphBuilderManager(pyus)
        self.phase = "init"
        self.node_id = 0
        self.spu_config = spu_config

    def convert_comp(
        self, param: NodeEvalParam, in_ds: List[DistData], out_ds: List[DistData]
    ):
        if self.phase == "init":
            # init -> pre/ml
            assert param.domain in ["preprocessing", "ml.train", "ml.predict"]
        elif self.phase == "preprocessing":
            # pre -> pre/ml
            assert param.domain in ["preprocessing", "ml.train", "ml.predict"]
        elif self.phase == "ml":
            # ml -> post
            # only support one ml in graph
            # not support postprocessing for now
            assert param.domain in ["postprocessing"]
        elif self.phase == "postprocessing":
            # post -> post
            # not support postprocessing for now
            assert param.domain in ["postprocessing"]

        self.node_id += 1
        if param.domain == "preprocessing":
            self.phase = "preprocessing"
            preprocessing_converter(
                ctx=self.ctx,
                builder=self.graph_builders,
                node_id=self.node_id,
                param=param,
                in_ds=in_ds,
                out_ds=out_ds,
            )
        elif param.domain in ["ml.train", "ml.predict"]:
            self.phase = "ml"
            train_model_converter(
                ctx=self.ctx,
                builder=self.graph_builders,
                node_id=self.node_id,
                spu_config=self.spu_config,
                param=param,
                in_ds=in_ds,
                out_ds=out_ds,
            )
        else:
            # TODO: postprocessing
            raise AttributeError(f"not support domain {param.domain}")

    def dump_tar_files(self, name, desc, uri) -> None:
        self.graph_builders.dump_tar_files(name, desc, self.ctx, uri)


@model_export_comp.eval_fn
def model_export_comp_fn(
    *,
    ctx,
    model_name,
    model_desc,
    input_datasets,
    output_datasets,
    component_eval_params,
    package_output,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    input_datasets = [json_format.Parse(i, DistData()) for i in input_datasets]
    output_datasets = [json_format.Parse(o, DistData()) for o in output_datasets]
    component_eval_params = [
        json_format.Parse(base64.b64decode(o).decode("utf-8"), NodeEvalParam())
        for o in component_eval_params
    ]

    assert input_datasets and output_datasets and component_eval_params
    # TODO: assert system_info
    system_info = input_datasets[0].system_info

    pyus = get_init_pyus(input_datasets, component_eval_params)
    builder = CompConverter(ctx, pyus, spu_config)

    for param in component_eval_params:
        comp_def = get_comp_def(param.domain, param.name, param.version)
        builder.convert_comp(
            param,
            input_datasets[: len(comp_def.inputs)],
            output_datasets[: len(comp_def.outputs)],
        )
        input_datasets = input_datasets[len(comp_def.inputs) :]
        output_datasets = output_datasets[len(comp_def.outputs) :]

    uid = reveal(pyus[0](lambda: str(uuid.uuid4()))())
    model_name = f"{model_name}_{uid}"

    builder.dump_tar_files(model_name, model_desc, package_output)

    model_dd = DistData(
        name=model_name,
        type=str(DistDataType.SERVING_MODEL),
        system_info=system_info,
        data_refs=[
            DistData.DataRef(uri=package_output, party=p.party, format="tar.gz")
            for p in pyus
        ],
    )

    return {"package_output": model_dd}
