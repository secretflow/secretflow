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
from secretflow_spec import ObjectFile, Output, Storage, Version, VTable, make_storage
from secretflow_spec.v1.data_pb2 import DistData, StorageConfig

from secretflow.component.core.dist_data.tarfile import TarFile
from secretflow.device.device.heu import HEU, heu_from_base_config
from secretflow.device.device.pyu import PYU
from secretflow.device.device.spu import SPU
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig
from secretflow.utils.errors import InvalidArgumentError, InvalidStateError

from .checkpoint import Checkpoint
from .config import extract_device_config
from .dataframe import CompVDataFrame
from .dist_data.base import IDumper
from .dist_data.model import Model
from .progressor import IProgressor, new_progressor
from .types import TimeTracer


@dataclass
class SPURuntimeConfig:
    field: str = None
    fxp_fraction_bits: int = None
    fxp_exp_mode: int = None
    fxp_exp_iters: int = None
    experimental_exp_prime_offset: int = None
    experimental_exp_prime_disable_lower_bound: bool = None
    experimental_exp_prime_enable_upper_bound: bool = None

    def update_config(self, runtime_config: dict) -> dict:
        self.set_value(runtime_config, "field")
        self.set_value(runtime_config, "fxp_fraction_bits")
        self.set_value(runtime_config, "fxp_exp_iters")

        if self.fxp_exp_mode is not None:
            runtime_config["fxp_exp_mode"] = self.fxp_exp_mode
            if self.fxp_exp_mode == 3:
                runtime_config["experimental_enable_exp_prime"] = True
                self.set_value(runtime_config, "experimental_exp_prime_offset")
                self.set_value(
                    runtime_config, "experimental_exp_prime_disable_lower_bound"
                )
                self.set_value(
                    runtime_config, "experimental_exp_prime_enable_upper_bound"
                )

        return runtime_config

    def set_value(self, runtime_config: dict, name: str):
        value = getattr(self, name)
        if value is not None:
            runtime_config[name] = value


class Context:
    def __init__(
        self,
        storage_config: StorageConfig,
        cluster_config: SFClusterConfig,
        checkpoint: Checkpoint,
    ) -> None:
        if storage_config.type == "local_fs":
            data_dir = storage_config.local_fs.wd
        else:
            # FIXME: is this ok ?
            self_party = cluster_config.private_config.self_party
            data_dir = os.path.join(os.getcwd(), f"data_{os.getpid()}_{self_party}")
        self.cluster_config = cluster_config
        self._checkpoint = checkpoint
        self._storage = make_storage(storage_config)
        self.data_dir = data_dir
        self._spu_configs, self._heu_config = extract_device_config(cluster_config)
        self.tracer = TimeTracer()
        self.initiator_party: str = None
        self._progressor: IProgressor = None
        if cluster_config and cluster_config.public_config.webhook_config:
            progress_url = cluster_config.public_config.webhook_config.progress_url
            self._progressor = new_progressor(progress_url)

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def parties(self) -> list[str]:
        return self.cluster_config.desc.parties

    @property
    def self_party(self) -> str:
        return self.cluster_config.private_config.self_party

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
        field_type: spu.FieldType = spu.FieldType.FM64,
        fxp_fraction_bits: int = 0,
    ) -> HEU:
        if self._heu_config is None:
            raise InvalidStateError("need heu config in SFClusterDesc")
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
        field_type: spu.FieldType = spu.FieldType.FM64,
        fxp_fraction_bits: int = 0,
    ) -> dict[str, HEU]:
        if self._heu_config is None:
            raise InvalidStateError("need heu config in SFClusterDesc")
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

    def make_spu(self, config: SPURuntimeConfig = None) -> SPU:
        if not self._spu_configs:
            raise InvalidStateError("spu config is not found.")

        spu_config = next(iter(self._spu_configs.values()))
        return self._make_spu(spu_config, config)

    def make_spus(self, config: SPURuntimeConfig = None) -> list[SPU]:
        if not self._spu_configs:
            raise InvalidStateError("spu config is not found.")
        result = []
        for spu_config in self._spu_configs.values():
            spu = self._make_spu(spu_config, config)
            result.append(spu)
        return result

    @staticmethod
    def _make_spu(spu_config, config: SPURuntimeConfig | None) -> SPU:
        if config is not None:
            cluster_def = spu_config["cluster_def"].copy()
            config.update_config(cluster_def["runtime_config"])
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
        dd: DistData | ObjectFile,
        model_type: str = None,
        version: Version = None,
        pyus: dict[str, PYU] = None,
        spu: SPU = None,
    ) -> Model:
        if isinstance(dd, DistData):
            dd = ObjectFile.from_distdata(dd)

        dd.check(model_type, version)
        with self.trace_io():
            spu = spu if spu else lambda: self.make_spu()
            return Model.load(self.storage, dd, pyus=pyus, spu=spu)

    def load_tarfile(
        self,
        dd: DistData | ObjectFile,
        file_type: str = None,
        version: Version = None,
        base_dir: str = None,
    ) -> TarFile:
        if isinstance(dd, DistData):
            dd = ObjectFile.from_distdata(dd)
        dd.check(file_type, version)

        if base_dir is None:
            base_dir = self.data_dir
        with self.trace_io():
            return TarFile.load(self.storage, dd, base_dir)

    def dump(self, obj: IDumper, uri: str) -> DistData:
        with self.trace_io():
            return obj.dump(self.storage, uri)

    def dump_to(self, obj: IDumper, out: Output):
        if out.uri == "":
            raise InvalidArgumentError("output uri cannot be empty")

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
            dd = ObjectFile.from_distdata(payload)
            dd.check(model_type, version)

            spu = spu if spu else lambda: self.make_spu()
            model = Model.load(self.storage, dd, pyus=pyus, spu=spu)
            return model

    def dump_checkpoint(self, step: int, model: Model, model_uri: str):
        if not self.enable_checkpoint:
            return
        with self.trace_io():
            payload = model.dump(self.storage, model_uri)
            self._checkpoint.dump(self.storage, step, payload)

    def update_progress(self, percent: float, infos: dict | None = None):
        if self._progressor is not None:
            self._progressor.update(percent, infos)

    def on_finish(self):
        if self._progressor is not None:
            self._progressor.done()
