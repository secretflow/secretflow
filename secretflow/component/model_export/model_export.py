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
import logging
import uuid
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
from google.protobuf import json_format

from secretflow.component.component import CompEvalError, Component, IoType
from secretflow.component.data_utils import DistDataType, extract_data_infos
from secretflow.component.model_export.serving_utils.postprocessing_converter import (
    PostprocessingConverter,
)
from secretflow.device import PYU, reveal
from secretflow.spec.v1.data_pb2 import DistData
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report

from .serving_utils import (
    GraphBuilderManager,
    PreprocessingConverter,
    TrainModelConverter,
)

model_export_comp = Component(
    "model_export",
    domain="model",
    version="0.0.2",
    desc=(
        "The model_export component supports converting and "
        "packaging the rule files generated by preprocessing and "
        "postprocessing components, as well as the model files generated "
        "by model operators, into a Secretflow-Serving model package. The "
        "list of components to be exported must contain exactly one model "
        "train or model predict component, and may include zero or "
        "multiple preprocessing and postprocessing components."
    ),
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
    desc=(
        "The input data IDs for all components to be exported. "
        "Their order must remain consistent with the sequence in which the components were executed."
    ),
    is_list=True,
    is_optional=False,
)

model_export_comp.str_attr(
    name="output_datasets",
    desc=(
        "The output data IDs for all components to be exported. "
        "Their order must remain consistent with the sequence in which the components were executed."
    ),
    is_list=True,
    is_optional=False,
)

model_export_comp.str_attr(
    name="component_eval_params",
    desc=(
        "The eval parameters (in JSON format) for all components to be exported. "
        "Their order must remain consistent with the sequence in which the components were executed."
    ),
    is_list=True,
    is_optional=False,
)

model_export_comp.bool_attr(
    name="he_mode",
    desc=(
        "If enabled, it will export a homomorphic encryption model. Currently, only SGD and GLM models for two-party scenarios are supported."
    ),
    is_list=False,
    is_optional=False,
    default_value=False,
)

model_export_comp.io(
    io_type=IoType.OUTPUT,
    name="package_output",
    desc="output tar package uri",
    types=[DistDataType.SERVING_MODEL],
)

model_export_comp.io(
    io_type=IoType.OUTPUT,
    name="report",
    desc="report dumped model's input schemas",
    types=[DistDataType.REPORT],
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
    infos = extract_data_infos(
        v_tables[0], load_features=True, load_ids=True, load_labels=True
    )
    assert len(infos) > 0

    return [PYU(p) for p in infos], {p: infos[p].dtypes for p in infos}


class CompConverter:
    def __init__(
        self,
        ctx,
        pyus: List[PYU],
        spu_config,
        input_schemas: Dict[str, Dict[str, np.dtype]],
        heu_config,
        he_mode,
    ):
        self.ctx = ctx
        self.graph_builders = GraphBuilderManager(pyus)
        self.phase = "init"
        self.node_id = 0
        self.spu_config = spu_config
        self.heu_config = heu_config
        self.he_mode = he_mode
        self.input_schemas = {
            p: {c for c in v.keys()} for p, v in input_schemas.items()
        }
        self.converters = []
        self.schema_infos_cache = []
        self.used_schemas = defaultdict(set)
        self.deleted_schemas = defaultdict(set)
        self.derived_schemas = defaultdict(set)

    def _update_preprocessing_schema_info(
        self, schema_info: Dict[str, Dict[str, Set[str]]]
    ):
        for party in schema_info:
            used = schema_info[party]["used"]
            deleted = schema_info[party]["deleted"]
            derived = schema_info[party]["derived"]
            used = used - self.deleted_schemas[party]
            used = used - self.derived_schemas[party]
            self.used_schemas[party].update(used)
            self.deleted_schemas[party].update(deleted)
            self.derived_schemas[party] = self.derived_schemas[party] - deleted
            self.derived_schemas[party].update(derived)

    def _updata_train_schema_info(self, schema_info: Dict[str, List[str]]):
        for party in schema_info:
            used = schema_info[party]["used"]
            used = used - self.deleted_schemas[party]
            used = used - self.derived_schemas[party]
            self.used_schemas[party].update(used)
            assert party in self.input_schemas
            assert self.used_schemas[party].issubset(
                self.input_schemas[party]
            ), f"used_schemas: {self.used_schemas[party]}, input_schemas: {self.input_schemas[party]}"

        # schema trace is over
        del self.deleted_schemas
        del self.derived_schemas

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
            converter = PreprocessingConverter(
                ctx=self.ctx,
                builder=self.graph_builders,
                node_id=self.node_id,
                param=param,
                in_ds=in_ds,
                out_ds=out_ds,
            )
            schema_info = converter.schema_info()
            logging.info(
                f"comp:\n ====== \n{param}\n ====== \n"
                f"in_ds:\n ====== \n{in_ds}\n ====== \n"
                f"out_ds: \n ====== \n{out_ds}\n ====== \n"
                f"schema_info:\n ====== \n{schema_info}\n ====== \n"
            )
            self._update_preprocessing_schema_info(schema_info)
            self.converters.append(converter)
            self.schema_infos_cache.append(schema_info)
        elif param.domain in ["ml.train", "ml.predict"]:
            self.phase = "ml"
            converter = TrainModelConverter(
                ctx=self.ctx,
                builder=self.graph_builders,
                node_id=self.node_id,
                spu_config=self.spu_config,
                heu_config=self.heu_config,
                he_mode=self.he_mode,
                param=param,
                in_ds=in_ds,
                out_ds=out_ds,
            )
            schema_info = converter.schema_info()
            logging.info(
                f"comp:\n ====== \n{param}\n ====== \n"
                f"in_ds:\n ====== \n{in_ds}\n ====== \n"
                f"out_ds: \n ====== \n{out_ds}\n ====== \n"
                f"schema_info:\n ====== \n{schema_info}\n ====== \n"
            )
            self._updata_train_schema_info(schema_info)
            self.converters.append(converter)
            self.schema_infos_cache.append(schema_info)
        elif param.domain == "postprocessing":
            self.phase = "postprocessing"
            converter = PostprocessingConverter(
                ctx=self.ctx,
                builder=self.graph_builders,
                node_id=self.node_id,
                param=param,
                in_ds=in_ds,
                out_ds=out_ds,
            )
            self.converters.append(converter)
        else:
            raise AttributeError(f"not support domain {param.domain}")

    def dump_tar_files(self, name, desc, uri) -> None:
        assert self.phase in ["ml", "postprocessing"]
        traced_input = self.used_schemas
        for i, c in enumerate(self.converters):
            if isinstance(c, PostprocessingConverter):
                c.convert()
                continue
            schema_info = self.schema_infos_cache[i]
            for party in traced_input:
                assert party in schema_info
                assert schema_info[party]["used"].issubset(traced_input[party])

            traced_output = dict()
            if isinstance(c, PreprocessingConverter):
                for party in traced_input:
                    traced_output[party] = (
                        traced_input[party] - schema_info[party]["deleted"]
                    )
                    traced_output[party].update(schema_info[party]["derived"])

            c.convert(traced_input, traced_output)
            traced_input = traced_output

        self.graph_builders.dump_tar_files(name, desc, self.ctx, uri)


@model_export_comp.eval_fn
def model_export_comp_fn(
    *,
    ctx,
    model_name,
    model_desc,
    input_datasets,
    output_datasets,
    he_mode,
    component_eval_params,
    package_output,
    report,
):
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    if he_mode and ctx.heu_config is None:
        raise CompEvalError("heu config is not found while he_mode is True")

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

    pyus, complete_schemas = get_init_pyus(input_datasets, component_eval_params)
    builder = CompConverter(
        ctx, pyus, spu_config, complete_schemas, ctx.heu_config, he_mode
    )

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

    used = builder.used_schemas

    report_mate = Report(
        name="used schemas",
        desc=",".join([s for ss in used.values() for s in ss]),
    )

    report_dd = DistData(
        name=report,
        type=str(DistDataType.REPORT),
        system_info=system_info,
    )
    report_dd.meta.Pack(report_mate)

    return {"package_output": model_dd, "report": report_dd}
