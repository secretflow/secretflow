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


import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

from secretflow.spec.v1.data_pb2 import DistData

from ..common.types import BaseEnum
from ..storage import Storage


@enum.unique
class DistDataType(BaseEnum):
    # tables
    VERTICAL_TABLE = "sf.table.vertical_table"
    INDIVIDUAL_TABLE = "sf.table.individual"
    # models
    SS_SGD_MODEL = "sf.model.ss_sgd"
    SS_GLM_MODEL = "sf.model.ss_glm"
    SGB_MODEL = "sf.model.sgb"
    SS_XGB_MODEL = "sf.model.ss_xgb"
    SL_NN_MODEL = "sf.model.sl_nn"
    # binning rule
    BINNING_RULE = "sf.rule.binning"
    # others preprocessing rules
    PREPROCESSING_RULE = "sf.rule.preprocessing"
    # report
    REPORT = "sf.report"
    # read data
    READ_DATA = "sf.read_data"
    # serving model file
    SERVING_MODEL = "sf.serving.model"
    # checkpoints
    SS_GLM_CHECKPOINT = "sf.checkpoint.ss_glm"
    SGB_CHECKPOINT = "sf.checkpoint.sgb"
    SS_XGB_CHECKPOINT = "sf.checkpoint.ss_xgb"
    SS_SGD_CHECKPOINT = "sf.checkpoint.ss_sgd"
    # unbalance psi
    UNBALANCE_PSI_CACHE = "sf.model.ub_psi.cache"
    # if input of component is optional, then the corresponding type can be NULL
    NULL = "sf.null"


class IDumper(ABC):
    @abstractmethod
    def dump(self, storage: Storage, uri: str, **kwargs) -> DistData:  # type: ignore
        raise NotImplementedError(f"dump not implemented {uri}")


@dataclass
class Version:
    major: int
    minor: int

    def check(self, max_version: 'Version'):
        if max_version and not (
            max_version.major == self.major and max_version.minor >= self.minor
        ):
            raise ValueError(f"model version mismatch, {self}, {max_version}")
