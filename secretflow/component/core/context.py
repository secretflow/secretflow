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


import os
from dataclasses import dataclass

import spu

from secretflow.device.device.heu import HEU, heu_from_base_config
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU
from secretflow.error_system.exceptions import (
    EvalParamError,
    SFTrainingHyperparameterError,
)
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig
from secretflow.spec.v1.data_pb2 import DistData, StorageConfig

from .checkpoint import Checkpoint
from .common.types import Output, TimeTracer
from .config import extract_device_config
from .dist_data.base import IDumper
from .dist_data.model import Model, Version
from .dist_data.vtable import CompVDataFrame, VTable
from .storage import Storage


@dataclass
class SPURuntimeConfig:
    field: str | None = None
    fxp_fraction_bits: int | None = None
    fxp_exp_mode: int | None = None
    fxp_exp_iters: int | None = None


class Context:
    def __init__(
        self, storage_config: StorageConfig, cluster_config: SFClusterConfig, checkpoint: Checkpoint  # type: ignore
    ) -> None:
        if storage_config.type == "local_fs":
            data_dir = storage_config.local_fs.wd
        else:
            # FIXME: is this ok ?
            self_party = cluster_config.private_config.self_party
            data_dir = os.path.join(os.getcwd(), f"data_{os.getpid()}_{self_party}")
        self.cluster_config = cluster_config
        self._checkpoint = checkpoint
        self._storage = Storage(storage_config)
        self.data_dir = data_dir
        self._spu_configs, self._heu_config = extract_device_config(cluster_config)
        self.tracer = TimeTracer()
        self.initiator_party: str | None = None

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def parties(self) -> list[str]:
        return self.cluster_config.desc.parties

    def trace_running(self):
        return self.tracer.trace_running()

    def trace_io(self):
        return self.tracer.trace_io()

    def trace_report(self) -> dict:
        return self.tracer.report()

    def make_heu(
        self,
        new_sk_keeper: str,
        new_evaluators: list[str],
        field_type: spu.spu_pb2.FieldType = spu.spu_pb2.FM64,
        fxp_fraction_bits: int = 0,
    ) -> HEU:
        if self._heu_config is None:
            raise SFTrainingHyperparameterError.sf_cluster_config_error(
                "need heu config in SFClusterDesc"
            )
        return heu_from_base_config(
            self._heu_config,
            new_sk_keeper,
            new_evaluators,
            field_type,
            fxp_fraction_bits,
        )

    def make_heus(
        self,
        parties: list[str],
        field_type: spu.spu_pb2.FieldType = spu.spu_pb2.FM64,
        fxp_fraction_bits: int = 0,
    ) -> dict[str, HEU]:
        assert self._heu_config is not None, "need heu config in SFClusterDesc"
        heu_dict = {}
        for party in parties:
            heu = heu_from_base_config(
                self._heu_config,
                party,
                [p for p in parties if p != party],
                field_type,
                fxp_fraction_bits,
            )
            heu_dict[party] = heu
        return heu_dict

    def make_spu(self, config: SPURuntimeConfig | None = None) -> SPU:
        if self._spu_configs is None or len(self._spu_configs) == 0:
            raise SFTrainingHyperparameterError.sf_cluster_config_error(
                "spu config is not found."
            )
        spu_config = next(iter(self._spu_configs.values()))
        return self._make_spu(spu_config, config)

    def make_spus(self, config: SPURuntimeConfig | None = None) -> list[SPU]:
        if self._spu_configs is None or len(self._spu_configs) == 0:
            raise SFTrainingHyperparameterError.sf_cluster_config_error(
                "spu config is not found."
            )
        result = []
        for spu_config in self._spu_configs.values():
            spu = self._make_spu(spu_config, config)
            result.append(spu)
        return result

    @staticmethod
    def _make_spu(spu_config, config: SPURuntimeConfig | None) -> SPU:
        if config is not None:
            cluster_def = spu_config["cluster_def"].copy()
            runtime_config = cluster_def["runtime_config"]
            if config.field is not None:
                runtime_config["field"] = config.field
            if config.fxp_fraction_bits is not None:
                runtime_config["fxp_fraction_bits"] = config.fxp_fraction_bits
            if config.fxp_exp_mode is not None:
                runtime_config["fxp_exp_mode"] = config.fxp_exp_mode
            if config.fxp_exp_iters is not None:
                runtime_config["fxp_exp_iters"] = config.fxp_exp_iters
        else:
            cluster_def = spu_config["cluster_def"]
        return SPU(cluster_def, spu_config["link_desc"])

    def load_table(
        self, dd: DistData | VTable, columns: list[str] | None = None
    ) -> CompVDataFrame:
        if columns:
            if isinstance(dd, DistData):
                dd = VTable.from_distdata(dd, columns)
            else:
                dd = dd.select(columns)
        with self.trace_io():
            return CompVDataFrame.load(self.storage, dd)

    def load_model(
        self,
        dd: DistData,
        model_type: str = None,
        version: Version = None,
        pyus: dict[str, PYU] = None,
        spu: SPU = None,
    ) -> Model:
        with self.trace_io():
            spu = spu if spu else lambda: self.make_spu()
            model = Model.load(self.storage, dd, pyus=pyus, spu=spu)
            model.check(model_type, version)
            return model

    def dump(self, obj: IDumper, uri: str) -> DistData:
        with self.trace_io():
            return obj.dump(self.storage, uri)

    def dump_to(self, obj: IDumper, out: Output):
        if out.uri == "":
            raise EvalParamError("output uri cannot be empty")
        with self.trace_io():
            out.data = obj.dump(self.storage, out.uri)

    @property
    def enable_checkpoint(self) -> bool:
        return self._checkpoint is not None

    def load_checkpoint(
        self,
        model_type: str = None,
        version: Version = None,
        pyus: dict[str, PYU] = None,
        spu: SPU = None,
    ) -> Model | None:
        if not self.enable_checkpoint:
            return None
        with self.trace_io():
            payload = self._checkpoint.load(self.storage)
            if payload is None:
                return None
            spu = spu if spu else lambda: self.make_spu()
            model = Model.load(self.storage, payload, pyus=pyus, spu=spu)
            model.check(model_type, version)
            return model

    def dump_checkpoint(self, step: int, model: Model, model_uri: str):
        if not self.enable_checkpoint:
            return
        with self.trace_io():
            payload = model.dump(self.storage, model_uri)
            self._checkpoint.dump(self.storage, step, payload)
