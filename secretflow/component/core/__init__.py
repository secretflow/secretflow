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

from secretflow_spec import *

from .connector import new_connector
from .context import Context, SPURuntimeConfig
from .dataframe import (
    CompPartition,
    CompVDataFrame,
    CompVDataFrameReader,
    CompVDataFrameWriter,
    save_prediction,
)
from .dist_data.base import DistDataType
from .dist_data.model import Model
from .dist_data.tarfile import TarFile
from .dist_data.vtable_utils import VTableUtils
from .entry import comp_eval, format_exception
from .envs import Envs, get_bool_env, get_env
from .i18n import Translator, get_translation, translate
from .io import (
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
from .plugin import Plugin, PluginManager, load_plugins
from .resources import Resources, ResourceType
from .serving_builder import (
    DispatchType,
    IServingExporter,
    ServingBuilder,
    ServingNode,
    ServingOp,
    ServingPhase,
)
from .types import PathCleanUp, TimeTracer
from .utils import (
    assert_almost_equal,
    download_csv,
    download_files,
    float_almost_equal,
    get_comp_list_def,
    pad_inf_to_split_points,
    upload_files,
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
SS_KMEANS_MODEL_MAX = Version(0, 1)
SS_GPC_MODEL_MAX = Version(0, 1)
SS_GNB_MODEL_MAX = Version(0, 1)
SS_KNN_MODEL_MAX = Version(0, 1)
SGB_MODEL_MAX = Version(0, 1)
UB_PSI_CACHE_MAX = Version(0, 1)

SPU_RUNTIME_CONFIG_FM128_FXP40 = SPURuntimeConfig(field="FM128", fxp_fraction_bits=40)

__all__ = [
    # connector
    "new_connector",
    # entry
    "comp_eval",
    "format_exception",
    # context
    "Context",
    "SPURuntimeConfig",
    # dataframe
    "CompPartition",
    "CompVDataFrame",
    "CompVDataFrameReader",
    "CompVDataFrameWriter",
    "save_prediction",
    # dist_data
    "DistDataType",
    "Model",
    "TarFile",
    "VTableUtils",
    # envs
    "Envs",
    "get_bool_env",
    "get_env",
    # i18n
    "Translator",
    "get_translation",
    "translate",
    # io
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
    # plugin
    "Plugin",
    "PluginManager",
    "load_plugins",
    # resources
    "Resources",
    "ResourceType",
    # serving_builder
    "DispatchType",
    "IServingExporter",
    "ServingBuilder",
    "ServingNode",
    "ServingOp",
    "ServingPhase",
    # types
    "PathCleanUp",
    "TimeTracer",
    # utils
    "assert_almost_equal",
    "download_csv",
    "download_files",
    "float_almost_equal",
    "get_comp_list_def",
    "pad_inf_to_split_points",
    "upload_files",
    "upload_orc",
    "uuid4",
]
