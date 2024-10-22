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

from io import BufferedIOBase
from typing import Dict

from secretflow.spec.v1.data_pb2 import StorageConfig

from .impl import BuildStorageImpl


class ComponentStorage:
    def __init__(self, config: StorageConfig) -> None:
        # make sure ComponentStorage is always pickle-able
        # Only reference pb config in this class
        self._config = config

    def download_file(self, remote_fn, local_fn) -> None:
        """blocked download whole file into local_fn"""
        impl = BuildStorageImpl(self._config)
        impl.download_file(remote_fn, local_fn)

    def upload_file(self, remote_fn, local_fn) -> None:
        """blocked upload whole file from local_fn"""
        impl = BuildStorageImpl(self._config)
        impl.upload_file(remote_fn, local_fn)

    def get_reader(self, remote_fn) -> BufferedIOBase:
        impl = BuildStorageImpl(self._config)
        return impl.get_reader(remote_fn)

    def get_writer(self, remote_fn) -> BufferedIOBase:
        impl = BuildStorageImpl(self._config)
        return impl.get_writer(remote_fn)

    def get_file_meta(self, remote_fn) -> Dict:
        impl = BuildStorageImpl(self._config)
        return impl.get_file_meta(remote_fn)

    def remove(self, remote_fn) -> None:
        impl = BuildStorageImpl(self._config)
        return impl.remove(remote_fn)

    def exists(self, remote_fn) -> bool:
        impl = BuildStorageImpl(self._config)
        return impl.exists(remote_fn)
