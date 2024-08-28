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

from secretflow.device.device.heu import HEU, heu_from_base_config
from secretflow.device.device.spu import SPU
from secretflow.spec.extend.cluster_pb2 import SFClusterConfig
from secretflow.spec.v1.data_pb2 import DistData, StorageConfig

from .checkpoint import Checkpoint
from .common.types import Output, TimeTracer
from .config import extract_device_config
from .dist_data.base import IDumper
from .dist_data.model import Model
from .dist_data.vtable import CompVDataFrame, VTableFieldKind
from .storage import Storage


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

        self._checkpoint = checkpoint
        self._storage = Storage(storage_config)
        self._data_dir = data_dir
        self._spu_configs, self._heu_config = extract_device_config(cluster_config)
        self.tracer = TimeTracer()

    @property
    def storage(self) -> Storage:
        return self._storage

    def trace_running(self):
        return self.tracer.trace_running()

    def trace_io(self):
        return self.tracer.trace_io()

    def trace_report(self) -> dict:
        return self.tracer.report()

    def make_heu(self, new_sk_keeper: str, new_evaluators: list[str]) -> HEU:
        assert self._heu_config is not None, "need heu config in SFClusterDesc"
        return heu_from_base_config(self._heu_config, new_sk_keeper, new_evaluators)

    def make_spu(self) -> SPU:
        assert (
            self._spu_configs is not None and len(self._spu_configs) > 0
        ), "spu config is not found."
        spu_config = next(iter(self._spu_configs.values()))
        return self._make_spu(spu_config)

    def make_spus(self) -> list[SPU]:
        assert (
            self._spu_configs is not None and len(self._spu_configs) > 0
        ), "spu config is not found."
        result = []
        for spu_config in self._spu_configs.values():
            spu = self._make_spu(spu_config)
            result.append(spu)
        return result

    @staticmethod
    def _make_spu(spu_config) -> SPU:
        cluster_def = spu_config["cluster_def"].copy()
        # forced to use 128 ring size & 40 fxp
        cluster_def["runtime_config"]["field"] = "FM128"
        cluster_def["runtime_config"]["fxp_fraction_bits"] = 40
        return SPU(cluster_def, spu_config["link_desc"])

    def load_table(
        self,
        dd: DistData,
        kinds: VTableFieldKind = VTableFieldKind.ALL,
        col_selects: list[str] = None,  # if None, load all cols
        partitions_order: list[str] = None,
    ) -> CompVDataFrame:  # type: ignore
        with self.trace_io():
            return CompVDataFrame.load(
                self.storage, dd, kinds, col_selects, partitions_order
            )

    def load_model(self, dd: DistData) -> Model:  # type: ignore
        with self.trace_io():
            return Model.load(self.storage, dd, spu=lambda: self.make_spu())

    def dump(self, obj: IDumper, uri: str) -> DistData:  # type: ignore
        with self.trace_io():
            return obj.dump(self.storage, uri)

    def dump_to(self, obj: IDumper, out: Output):
        assert out.uri != "", "output uri cannot be empty"
        with self.trace_io():
            out.data = obj.dump(self.storage, out.uri)

    @property
    def enable_checkpoint(self) -> bool:
        return self._checkpoint is not None

    def load_checkpoint(self) -> Model:
        with self.trace_io():
            payload = self._checkpoint.load(self.storage)
            return Model.load(self.storage, payload)

    def dump_checkpoint(self, step: int, model: Model, model_uri: str):
        if not self.enable_checkpoint:
            return
        with self.trace_io():
            payload = model.dump(self.storage, model_uri)
            self._checkpoint.dump(self.storage, step, payload)
