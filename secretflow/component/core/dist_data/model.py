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

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

from secretflow_spec import ObjectFile, Storage, Version
from secretflow_spec.v1.data_pb2 import DistData, SystemInfo

from secretflow.device.driver import PYU, SPU, DeviceObject, PYUObject, SPUObject, wait
from secretflow.utils import secure_pickle as pickle
from secretflow.utils.errors import InvalidStateError

from .base import IDumper


@dataclass
class _DeviceObjectItem:
    type: str
    data_ref_idxs: list[int]

    @staticmethod
    def from_dict(v: dict) -> '_DeviceObjectItem':
        return _DeviceObjectItem(v["type"], v["data_ref_idxs"])

    def to_dict(self) -> dict:
        return {"type": self.type, "data_ref_idxs": self.data_ref_idxs}


_DEVICE_OBJECTS_KEY = "device_objects"


class Model(IDumper):
    def __init__(
        self,
        name: str,
        type: str,
        version: Version = None,
        objs: list[DeviceObject] = None,
        public_info: Any = None,
        attributes: dict = None,
        system_info: SystemInfo = None,
    ):
        self.name = name
        self.type = str(type)
        self.version = version
        self.objs = objs
        self.attributes = attributes
        self.public_info = public_info
        self.system_info = system_info

    @property
    def pyu_objs(self) -> dict[PYU, PYUObject]:
        res = {}
        for obj in self.objs:
            assert isinstance(obj, PYUObject)
            res[obj.device] = obj

        return res

    @staticmethod
    def load(
        storage: Storage,
        dd: ObjectFile,
        pyus: dict[str, PYU] = None,
        spu: SPU | Callable[[], SPU] = None,
    ) -> 'Model':
        if not dd.attributes or _DEVICE_OBJECTS_KEY not in dd.attributes:
            raise ValueError(f"{_DEVICE_OBJECTS_KEY} not in {dd.attributes}")

        device_objects = json.loads(dd.attributes.pop(_DEVICE_OBJECTS_KEY))
        assert isinstance(device_objects, list)

        objs = []
        for save_obj_dict in device_objects:
            save_obj = _DeviceObjectItem.from_dict(save_obj_dict)
            if save_obj.type == "pyu":
                if len(save_obj.data_ref_idxs) != 1:
                    raise InvalidStateError("len of data_ref_idxs of pyu should be 1")

                data_ref = dd.data_refs[save_obj.data_ref_idxs[0]]
                party = data_ref.party
                if pyus is not None and party not in pyus:
                    raise InvalidStateError(
                        f"party<{party}> not in pyus<{pyus.keys()}>",
                    )

                if data_ref.format != "pickle":
                    raise InvalidStateError(
                        "format of data_ref in dist_data should be 'pickle'"
                    )

                def loads(storage: Storage, path: str) -> Any:
                    with storage.get_reader(path) as r:
                        return pickle.load(r)

                objs.append(PYU(party)(loads)(storage, data_ref.uri))
            elif save_obj.type == "spu":
                # TODO: only support one spu for now
                if spu is None:
                    raise InvalidStateError(
                        "input spu should not be None when loading model of type spu"
                    )
                if inspect.isfunction(spu):
                    spu = spu()
                if len(save_obj.data_ref_idxs) <= 1:
                    raise InvalidStateError(
                        "len of data_ref_idxs of spu in model_meta should be larger than 1"
                    )
                full_paths = {}
                for data_ref_idx in save_obj.data_ref_idxs:
                    data_ref = dd.data_refs[data_ref_idx]
                    if data_ref.format != "pickle":
                        raise InvalidStateError(
                            "format of dist_data.data_ref should be 'pickle'"
                        )
                    party = data_ref.party
                    if party in full_paths:
                        raise InvalidStateError(
                            f"found duplicated party {party} in dist_data.data_refs"
                        )
                    uri = data_ref.uri
                    full_paths[party] = lambda uri=uri: storage.get_reader(uri)
                if set(full_paths.keys()) != set(spu.actors.keys()):
                    raise InvalidStateError(
                        f"party of dist_data.data_refs not match with spu.actors, "
                        f"dist_data.data_refs: {set(full_paths.keys())}, spu.actors: {set(spu.actors.keys())}"
                    )
                spu_paths = [full_paths[party] for party in spu.actors.keys()]
                objs.append(spu.load(spu_paths))
            else:
                raise InvalidStateError(
                    f"not supported type {save_obj.type} in model_meta.objs"
                )

        return Model(
            name=dd.name,
            type=dd.type,
            version=dd.version,
            public_info=dd.public_info,
            attributes=dd.attributes,
            system_info=dd.system_info,
            objs=objs,
        )

    def dump(self, storage: Storage, output_uri: str) -> DistData:
        assert output_uri, f"output_uri cannot be empty"

        objs_uri = []
        objs_party = []
        saved_objs: list[_DeviceObjectItem] = []
        for i, obj in enumerate(self.objs):
            if isinstance(obj, PYUObject):
                device: PYU = obj.device
                uri = f"{output_uri}/{i}"

                def dumps(comp_storage, uri: str, obj: Any):
                    with comp_storage.get_writer(uri) as w:
                        pickle.dump(obj, w)

                wait(device(dumps)(storage, uri, obj))

                saved_obj = _DeviceObjectItem(type="pyu", data_ref_idxs=[len(objs_uri)])
                saved_objs.append(saved_obj)
                objs_uri.append(uri)
                objs_party.append(device.party)
            elif isinstance(obj, SPUObject):
                device: SPU = obj.device
                uris = [f"{output_uri}/{i}" for _ in device.actors]

                device.dump(
                    obj,
                    [lambda uri=uri: storage.get_writer(uri) for uri in uris],
                )

                data_ref_idxs = [len(objs_uri) + p for p in range(len(uris))]
                saved_obj = _DeviceObjectItem(type="spu", data_ref_idxs=data_ref_idxs)
                saved_objs.append(saved_obj)
                objs_uri.extend(uris)
                objs_party.extend(list(device.actors.keys()))
            else:
                raise InvalidStateError(f"not supported objs type {type(obj)}")

        data_refs = [
            DistData.DataRef(uri=uri, party=p, format="pickle")
            for uri, p in zip(objs_uri, objs_party)
        ]

        attributes = {
            _DEVICE_OBJECTS_KEY: json.dumps([obj.to_dict() for obj in saved_objs])
        }
        if self.attributes:
            attributes.update(self.attributes)

        dd = ObjectFile(
            name=self.name,
            type=self.type,
            data_refs=data_refs,
            version=self.version,
            public_info=self.public_info,
            attributes=attributes,
            system_info=self.system_info,
        )
        return dd.to_distdata()
