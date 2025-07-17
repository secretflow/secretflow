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
import tarfile

from secretflow_spec import ObjectFile, Storage, Version
from secretflow_spec.v1.data_pb2 import DistData, SystemInfo

from secretflow.device import PYU, reveal, wait

from .base import IDumper


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
    def load(storage: Storage, dd: ObjectFile, base_dir: str = "/tmp") -> 'TarFile':
        def load_fn(storage: Storage, uri: str) -> list[str]:
            with storage.get_reader(uri) as r, tarfile.open(
                fileobj=r, mode='r:gz'
            ) as tar:
                assert isinstance(tar, tarfile.TarFile)
                tar.extractall(base_dir)
                return [os.path.join(m.name) for m in tar.getmembers()]

        parties = []
        objs = []
        for dr in dd.data_refs:
            assert dr.format == "tar.gz"
            pyu = PYU(dr.party)
            objs.append(pyu(load_fn)(storage, dr.uri))
            parties.append(dr.party)
        obj_files = reveal(objs)
        files = {party: obj for party, obj in zip(parties, obj_files)}

        return TarFile(
            dd.name,
            dd.type,
            dd.version,
            files,
            dd.public_info,
            dd.system_info,
        )

    def dump(self, storage: Storage, output_uri: str) -> DistData:
        assert output_uri, f"output_uri cannot be empty"

        def dump_fn(storage: Storage, uri: str, files: list[str]):
            with storage.get_writer(uri) as w, tarfile.open(
                fileobj=w, mode='w:gz'
            ) as tar:
                for f in files:
                    archive_name = os.path.basename(f)
                    tar.add(f, arcname=archive_name, recursive=True)

        objs_uri = []
        objs_party = []
        dump_res = []
        for party, files in self.files.items():
            assert len(files) > 0
            res = PYU(party)(dump_fn)(storage, output_uri, files)
            dump_res.append(res)
            objs_uri.append(output_uri)
            objs_party.append(party)

        wait(dump_res)

        data_refs = [
            DistData.DataRef(uri=output_uri, party=p, format="tar.gz")
            for p in objs_party
        ]

        dd = ObjectFile(
            name=self.name,
            type=self.type,
            data_refs=data_refs,
            version=self.version,
            public_info=self.public_info,
            system_info=self.system_info,
        )
        return dd.to_distdata()
