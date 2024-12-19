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


from .common.io import (
    BufferedIO,
    CSVReader,
    CSVReadOptions,
    CSVWriteOptions,
    CSVWriter,
    IReader,
    IWriter,
    ORCReader,
    ORCReadOptions,
    ORCWriteOptions,
    ORCWriter,
    ReadOptions,
    WriteOptions,
    convert_io,
    read_csv,
    read_orc,
    write_csv,
    write_orc,
)
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
from .connector import new_connector
from .context import Context, SPURuntimeConfig
from .dataframe import (
    CompPartition,
    CompVDataFrame,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    save_prediction,
)
from .definition import Definition, Field, Interval
from .dist_data.base import DistDataType, download_files, upload_files
from .dist_data.model import Model, Version
from .dist_data.report import Reporter
from .dist_data.tarfile import TarFile
from .dist_data.vtable import (
    VTable,
    VTableField,
    VTableFieldKind,
    VTableFieldType,
    VTableFormat,
    VTableParty,
    VTableSchema,
)
from .envs import Envs, get_bool_env, get_env
from .i18n import Translator, get_translation, translate
from .plugin import Plugin, PluginManager, load_component_modules, load_plugins
from .registry import Registry, register
from .resources import Resources, ResourceType
from .serving_builder import (
    DispatchType,
    ServingBuilder,
    ServingNode,
    ServingOp,
    ServingPhase,
)
from .storage import Storage
from .utils import (
    PathCleanUp,
    assert_almost_equal,
    build_node_eval_param,
    clean_text,
    download_csv,
    float_almost_equal,
    pad_inf_to_split_points,
    upload_orc,
    uuid4,
)

BINNING_RULE_MAX = Version(0, 1)
PREPROCESSING_RULE_MAX = Version(0, 3)
GLM_MODEL_MAX = Version(0, 4)
SGB_MODEL_MAX = Version(0, 1)
SS_SGD_MODEL_MAX = Version(0, 1)
SS_GLM_MODEL_MAX = Version(0, 4)
SS_XGB_MODEL_MAX = Version(0, 1)
SGB_MODEL_MAX = Version(0, 1)
UB_PSI_CACHE_MAX = Version(0, 1)

SPU_RUNTIME_CONFIG_FM128_FXP40 = SPURuntimeConfig(field="FM128", fxp_fraction_bits=40)

__all__ = [
    "uuid4",
    "download_csv",
    "upload_orc",
    "float_almost_equal",
    "pad_inf_to_split_points",
    "assert_almost_equal",
    "build_node_eval_param",
    "clean_text",
    "register",
    "load_component_modules",
    "load_plugins",
    "download_files",
    "upload_files",
    "new_connector",
    "download_csv",
    "PathCleanUp",
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
    "TarFile",
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
    "BufferedIO",
    "CSVReader",
    "CSVReadOptions",
    "CSVWriteOptions",
    "CSVWriter",
    "IReader",
    "IWriter",
    "ORCReader",
    "ORCReadOptions",
    "ORCWriteOptions",
    "ORCWriter",
    "ReadOptions",
    "WriteOptions",
    "convert_io",
    "read_csv",
    "read_orc",
    "write_csv",
    "write_orc",
    "Plugin",
    "PluginManager",
    "ResourceType",
    "Resources",
    "translate",
    "get_translation",
    "Translator",
]
