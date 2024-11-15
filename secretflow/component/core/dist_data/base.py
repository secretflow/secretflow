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
import os
from abc import ABC, abstractmethod
from typing import Union

from secretflow.device.device.pyu import PYU
from secretflow.device.driver import wait
from secretflow.error_system.exceptions import (
    CompEvalError,
    SFTrainingHyperparameterError,
)
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
    # if input of component is optional, then the corresponding type can be NULL
    NULL = "sf.null"


class IDumper(ABC):
    @abstractmethod
    def dump(self, storage: Storage, uri: str, **kwargs) -> DistData:  # type: ignore
        raise NotImplementedError(f"dump not implemented {uri}")


def download_files(
    storage: Storage,
    remote_fns: dict[Union[str, PYU], str],
    local_fns: dict[Union[str, PYU], str],
    overwrite: bool = True,
):
    pyu_remotes = {
        p.party if isinstance(p, PYU) else p: remote_fns[p] for p in remote_fns
    }
    pyu_locals = {}
    for p in local_fns:
        k = p.party if isinstance(p, PYU) else p
        v = local_fns[p]
        pyu_locals[k] = v

    if set(pyu_remotes.keys()) != set(pyu_locals.keys()):
        raise CompEvalError.party_check_failed(
            f"pyu_remotes: [{pyu_remotes}] is not equal to pyu_locals: [{pyu_locals}]"
        )

    def download_file(storage, uri, output_path):
        if not overwrite and os.path.exists(output_path):
            # skip download
            if not os.path.isfile(output_path):
                raise SFTrainingHyperparameterError.not_a_file(
                    f"In download_file, when choosing not overwrite, the output_path {output_path} should be a file"
                )
        else:
            storage.download_file(uri, output_path)

    waits = []
    for p in pyu_remotes:
        remote_fn = pyu_remotes[p]
        local_fn = pyu_locals[p]
        waits.append(PYU(p)(download_file)(storage, remote_fn, local_fn))

    wait(waits)


def upload_files(
    storage: Storage,
    remote_fns: dict[Union[str, PYU], str],
    local_fns: dict[Union[str, PYU], str],
):
    pyu_remotes = {
        p.party if isinstance(p, PYU) else p: remote_fns[p] for p in remote_fns
    }
    pyu_locals = {p.party if isinstance(p, PYU) else p: local_fns[p] for p in local_fns}

    if set(pyu_remotes.keys()) != set(pyu_locals.keys()):
        raise CompEvalError.party_check_failed(
            f"pyu_remotes: [{pyu_remotes}] is not equal to pyu_locals: [{pyu_locals}]"
        )

    waits = []
    for p in pyu_remotes:
        waits.append(
            PYU(p)(lambda c, r, l: c.upload_file(r, l))(
                storage, pyu_remotes[p], pyu_locals[p]
            )
        )
    wait(waits)
