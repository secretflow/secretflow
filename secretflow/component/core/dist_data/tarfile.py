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

import json
import os
import tarfile

from secretflow.device import PYU, reveal, wait
from secretflow.error_system.exceptions import (
    DataFormatError,
    InvalidArgumentError,
    SFModelError,
)
from secretflow.spec.extend.data_pb2 import DeviceObjectCollection
from secretflow.spec.v1.data_pb2 import DistData, SystemInfo

from ..storage import Storage
from .base import IDumper, Version


class TarFile(IDumper):
    def __init__(
        self,
        name: str,
        type: str,
        version: Version,
        files: dict[str, list[str]],
        public_info: str | dict = None,
        system_info: SystemInfo = None,
    ):
        self.name = name
        self.type = type
        self.version = version
        self.files = files
        self.public_info = public_info
        self.system_info = system_info

    @staticmethod
    def load(storage: Storage, dist_data: DistData, max_version: Version = None, base_dir: str = "/tmp") -> 'TarFile':  # type: ignore
        model_meta = DeviceObjectCollection()
        if not dist_data.meta.Unpack(model_meta):
            raise DataFormatError.unpack_distdata_error(
                unpack_type="DeviceObjectCollection"
            )

        model_info = json.loads(model_meta.public_info)

        if not (
            isinstance(model_info, dict)
            and "major_version" in model_info
            and "minor_version" in model_info
            and "public_info" in model_info
        ):
            raise SFModelError.model_info_error(
                "model info format error, or model version not supported."
            )
        public_info = json.loads(model_info.pop("public_info"))
        major, minor = model_info.pop("major_version"), model_info.pop("minor_version")
        version = Version(major, minor)
        version.check(max_version)

        def load_fn(storage: Storage, uri: str) -> list[str]:
            with storage.get_reader(uri) as r, tarfile.open(
                fileobj=r, mode='r:gz'
            ) as tar:
                assert isinstance(tar, tarfile.TarFile)
                tar.extractall(base_dir)
                return [os.path.join(m.name) for m in tar.getmembers()]

        parties = []
        objs = []
        for save_obj in model_meta.objs:
            assert save_obj.type == "pyu"
            if len(save_obj.data_ref_idxs) != 1:
                raise SFModelError.model_info_error(
                    "len of data_ref_idxs of pyu should be 1"
                )
            dr = dist_data.data_refs[save_obj.data_ref_idxs[0]]
            assert dr.format == "tar.gz"
            party = dr.party
            pyu = PYU(party)
            objs.append(pyu(load_fn)(storage, dr.uri))
            parties.append(party)
        obj_files = reveal(objs)
        files = {party: obj for party, obj in zip(parties, obj_files)}

        return TarFile(
            dist_data.name,
            dist_data.type,
            version,
            files,
            public_info,
            dist_data.system_info,
        )

    def dump(self, storage: Storage, output_uri: str) -> DistData:  # type: ignore
        if output_uri == "":
            raise InvalidArgumentError(
                f"output_uris cannot be empty when dumping Model"
            )
        if self.name == "":
            raise ValueError(f"name of model is not set or empty, uri is {output_uri}")
        if self.type == "":
            raise ValueError(f"type is of model not set or empty, uri is {output_uri}")

        def dump_fn(storage: Storage, uri: str, files: list[str]):
            with storage.get_writer(uri) as w, tarfile.open(
                fileobj=w, mode='w:gz'
            ) as tar:
                for f in files:
                    archive_name = os.path.basename(f)
                    tar.add(f, arcname=archive_name, recursive=True)

        objs_uri = []
        objs_party = []
        saved_objs = []
        dump_res = []
        for party, files in self.files.items():
            assert len(files) > 0
            res = PYU(party)(dump_fn)(storage, output_uri, files)
            dump_res.append(res)
            saved_obj = DeviceObjectCollection.DeviceObject(
                type="pyu", data_ref_idxs=[len(objs_uri)]
            )
            saved_objs.append(saved_obj)
            objs_uri.append(output_uri)
            objs_party.append(party)

        wait(dump_res)

        public_info = json.dumps(self.public_info)

        model_info = {
            "major_version": self.version.major,
            "minor_version": self.version.minor,
            "public_info": public_info,
        }

        meta = DeviceObjectCollection(
            objs=saved_objs,
            public_info=json.dumps(model_info),
        )

        dd = DistData(
            name=self.name,
            type=str(self.type),
            system_info=self.system_info,
            data_refs=[
                DistData.DataRef(uri=output_uri, party=p, format="tar.gz")
                for p in objs_party
            ],
        )
        dd.meta.Pack(meta)
        return dd
