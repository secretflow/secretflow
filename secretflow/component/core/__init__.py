# Copyright 2024 Ant Group Co., Ltd.
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


from .common.types import (
    AutoNameEnum,
    BaseEnum,
    Input,
    Output,
    TimeTracer,
    UnionGroup,
    UnionSelection,
)
from .component import Component, TComponent
from .context import Context, SPURuntimeConfig
from .definition import Definition, Field, Interval
from .dist_data.base import DistDataType, download_files, upload_files
from .dist_data.model import Model, Version
from .dist_data.report import Reporter
from .dist_data.vtable import (
    CompPartition,
    CompVDataFrame,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    VTable,
    VTableField,
    VTableFieldKind,
    VTableFieldType,
    VTableFormat,
    VTableParty,
    VTableSchema,
    save_prediction,
)
from .envs import Envs, get_bool_env, get_env
from .plugin import load_component_modules, load_plugins
from .registry import Registry, register
from .serving_builder import (
    DispatchType,
    ServingBuilder,
    ServingNode,
    ServingOp,
    ServingPhase,
)
from .storage import Storage
from .utils import (
    build_node_eval_param,
    clean_text,
    float_almost_equal,
    gettext,
    pad_inf_to_split_points,
    uuid4,
)

BINNING_RULE_MAX = Version(0, 1)
PREPROCESSING_RULE_MAX = Version(0, 3)
GLM_MODEL_MAX = Version(0, 3)
SGB_MODEL_MAX = Version(0, 1)
SS_SGD_MODEL_MAX = Version(0, 1)
SS_GLM_MODEL_MAX = Version(0, 3)
SS_XGB_MODEL_MAX = Version(0, 1)
SGB_MODEL_MAX = Version(0, 1)

SPU_RUNTIME_CONFIG_FM128_FXP40 = SPURuntimeConfig(field="FM128", fxp_fraction_bits=40)

__all__ = [
    "uuid4",
    "float_almost_equal",
    "pad_inf_to_split_points",
    "build_node_eval_param",
    "gettext",
    "clean_text",
    "register",
    "load_component_modules",
    "load_plugins",
    "download_files",
    "upload_files",
    "Registry",
    "BaseEnum",
    "AutoNameEnum",
    "Input",
    "Output",
    "UnionGroup",
    "UnionSelection",
    "TimeTracer",
    "Component",
    "TComponent",
    "Field",
    "Interval",
    "Definition",
    "Context",
    "DispatchType",
    "ServingOp",
    "ServingPhase",
    "ServingNode",
    "ServingBuilder",
    "Storage",
    "DistDataType",
    "DistDataObject",
    "Version",
    "Model",
    "Reporter",
    "VTable",
    "VTableField",
    "VTableFieldKind",
    "VTableFieldType",
    "VTableFormat",
    "VTableSchema",
    "VTableParty",
    "CompPartition",
    "CompVDataFrame",
    "CompVDataFrameReader",
    "CompVDataFrameWriter",
    "save_prediction",
    "Envs",
    "get_env",
    "get_bool_env",
]
